"""Pure NumPy/stdlib Threes simulator.

The move mechanics intentionally mirror ``state_hunt.advance_line_toward_start``
and ``state_hunt.simulate_base_move``. A pair of 6144 tiles is a legal merge
into a terminal 12288 tile; no tile is inserted after that terminal merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable, Optional, Sequence

import numpy as np

Direction = int

DIRECTION_NAMES = ("up", "down", "left", "right")
UP, DOWN, LEFT, RIGHT = range(4)

BOARD_SIZE = 4
SMALL_BAG_SIZE = 12
LARGE_DELAY_PREVIEWS = 21
LARGE_SPAN_SMALLS = 20
BONUS_TRIGGER_TILE = 48
MAX_NORMAL_TILE = 6144
TERMINAL_TILE = 12288

TOKEN_EMPTY = "\u00b7"
TOKEN_BLUE = "\U0001f7e6"
TOKEN_RED = "\U0001f7e5"

SMALL_LABELS = ("red", "blue", "gray")
SMALL_TILE_VALUES = {"blue": 1, "red": 2, "gray": 3}
VALUE_TO_SMALL_LABEL = {value: label for label, value in SMALL_TILE_VALUES.items()}
SCORE_BY_VALUE = {
    0: 0,
    1: 0,
    2: 0,
    3: 3,
    6: 9,
    12: 27,
    24: 81,
    48: 243,
    96: 729,
    192: 2187,
    384: 6561,
    768: 19683,
    1536: 59049,
    3072: 177147,
    6144: 531441,
    12288: 1594323,
}


@dataclass(frozen=True)
class Preview:
    kind: str
    value: Optional[int]
    candidates: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        if self.kind == "bonus":
            return "large_candidates"
        return self.kind


@dataclass
class SimState:
    board: np.ndarray
    preview: Preview
    small_counts: dict[str, int]
    small_pos: int
    small_seen_total: int
    span_small_pos: int
    large_pending: bool
    max_tile: int
    move_count: int
    game_over: bool


@dataclass(frozen=True)
class StepInfo:
    moved: bool
    inserted_value: Optional[int]
    inserted_pos: Optional[tuple[int, int]]
    merge_score_delta: int
    score_delta: int
    eligible_positions: list[tuple[int, int]]
    terminal_merge: bool = False


@dataclass(frozen=True)
class PreviewOption:
    preview: Preview
    probability: float


def can_merge(a: int, b: int) -> bool:
    if a <= 0 or b <= 0:
        return False
    if {a, b} == {1, 2}:
        return True
    return a >= 3 and a == b and a < TERMINAL_TILE


def merge_value(a: int, b: int) -> int:
    if not can_merge(a, b):
        raise ValueError(f"Cannot merge {a} and {b}")
    if {a, b} == {1, 2}:
        return 3
    return a * 2


def advance_line_toward_start(line: Sequence[int]) -> tuple[list[int], bool]:
    cells = [int(v) for v in line]
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


def simulate_base_move(board: Sequence[Sequence[int]] | np.ndarray, action: Direction | str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    direction = direction_name(action)
    board_arr = np.asarray(board, dtype=np.int32)
    if board_arr.shape != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Expected a 4x4 board, got {board_arr.shape}")

    if direction in ("up", "down"):
        working = board_arr.T.tolist()
    else:
        working = board_arr.tolist()

    moved_rows: list[bool] = []
    result_rows: list[list[int]] = []
    for row in working:
        if direction in ("right", "down"):
            shifted, changed = advance_line_toward_start(list(reversed(row)))
            shifted = list(reversed(shifted))
        else:
            shifted, changed = advance_line_toward_start(row)
        result_rows.append(shifted)
        moved_rows.append(changed)

    if direction in ("up", "down"):
        board_after = np.asarray(result_rows, dtype=np.int32).T.copy()
    else:
        board_after = np.asarray(result_rows, dtype=np.int32)

    eligible_positions: list[tuple[int, int]] = []
    for idx, changed in enumerate(moved_rows):
        if not changed:
            continue
        if direction == "left" and int(board_after[idx, 3]) == 0:
            eligible_positions.append((idx, 3))
        elif direction == "right" and int(board_after[idx, 0]) == 0:
            eligible_positions.append((idx, 0))
        elif direction == "up" and int(board_after[3, idx]) == 0:
            eligible_positions.append((3, idx))
        elif direction == "down" and int(board_after[0, idx]) == 0:
            eligible_positions.append((0, idx))
    return board_after, eligible_positions


def direction_name(action: Direction | str) -> str:
    if isinstance(action, str):
        if action not in DIRECTION_NAMES:
            raise ValueError(f"Unsupported direction: {action}")
        return action
    idx = int(action)
    if idx < 0 or idx >= len(DIRECTION_NAMES):
        raise ValueError(f"Unsupported direction index: {action}")
    return DIRECTION_NAMES[idx]


def direction_index(direction: Direction | str) -> Direction:
    if isinstance(direction, str):
        return DIRECTION_NAMES.index(direction)
    return int(direction)


def score_tile(value: int) -> int:
    value = int(value)
    cached = SCORE_BY_VALUE.get(value)
    if cached is not None:
        return cached
    if value <= 2:
        return 0
    if value % 3 != 0:
        raise ValueError(f"Unsupported Threes tile value: {value}")
    exponent = log2(value // 3) + 1
    if int(exponent) != exponent:
        raise ValueError(f"Unsupported Threes tile value: {value}")
    return 3 ** int(exponent)


def score_board(board: Sequence[Sequence[int]] | np.ndarray) -> int:
    arr = np.asarray(board, dtype=np.int64)
    total = 0
    for value in arr.reshape(-1):
        value_int = int(value)
        total += SCORE_BY_VALUE.get(value_int, score_tile(value_int))
    return int(total)


def board_max_tile(board: Sequence[Sequence[int]] | np.ndarray) -> int:
    arr = np.asarray(board, dtype=np.int32)
    if arr.size == 0:
        return 0
    return int(arr.max(initial=0))


def board_to_tokens(board: Sequence[Sequence[int]] | np.ndarray) -> list[list[str]]:
    arr = np.asarray(board, dtype=np.int32)
    rows: list[list[str]] = []
    for row in arr.tolist():
        out_row: list[str] = []
        for value in row:
            if value <= 0:
                out_row.append(TOKEN_EMPTY)
            elif value == 1:
                out_row.append(TOKEN_BLUE)
            elif value == 2:
                out_row.append(TOKEN_RED)
            else:
                out_row.append(str(value))
        rows.append(out_row)
    return rows


def tokens_to_board(tokens: Sequence[Sequence[str]]) -> np.ndarray:
    rows: list[list[int]] = []
    for row in tokens:
        values: list[int] = []
        for token in row:
            if token == TOKEN_EMPTY:
                values.append(0)
            elif token == TOKEN_BLUE:
                values.append(1)
            elif token == TOKEN_RED:
                values.append(2)
            else:
                values.append(int(token))
        rows.append(values)
    return np.asarray(rows, dtype=np.int32)


def rank_for_value(value: int) -> int:
    value = int(value)
    if value <= 0:
        return 0
    if value in (1, 2):
        return value
    if value % 3 != 0:
        raise ValueError(f"Unsupported Threes tile value: {value}")
    power = log2(value // 3)
    if int(power) != power:
        raise ValueError(f"Unsupported Threes tile value: {value}")
    return int(power) + 3


def value_for_rank(rank: int) -> int:
    rank = int(rank)
    if rank <= 0:
        return 0
    if rank in (1, 2):
        return rank
    return 3 * (2 ** (rank - 3))


def preview_from_label(label: str, candidates: Iterable[int] = ()) -> Preview:
    if label == "large_candidates" or label == "bonus":
        cand_tuple = tuple(int(v) for v in candidates)
        return Preview(kind="bonus", value=None, candidates=cand_tuple)
    if label not in SMALL_TILE_VALUES:
        raise ValueError(f"Unsupported preview label: {label}")
    return Preview(kind=label, value=SMALL_TILE_VALUES[label])


def label_for_insert_value(value: int) -> str:
    return VALUE_TO_SMALL_LABEL.get(int(value), "large_candidates")


class ThreesSim:
    def __init__(self, rng: np.random.Generator, starter_tile: Optional[int] = 1536):
        self.rng = rng
        self.starter_tile = starter_tile

    def reset(self) -> SimState:
        small_counts = {"red": 4, "blue": 4, "gray": 4}
        small_pos = 0
        board_labels: list[str] = []
        for _ in range(8):
            label = self._sample_small_label(small_counts, small_pos)
            board_labels.append(label)
            if small_counts[label] <= 0:
                raise RuntimeError(f"Initial bag underflow for {label}")
            small_counts[label] -= 1
            small_pos += 1

        preview_label = self._sample_small_label(small_counts, small_pos)
        preview = preview_from_label(preview_label)

        values = [SMALL_TILE_VALUES[label] for label in board_labels]
        if self.starter_tile is not None:
            values.append(int(self.starter_tile))
        positions = self.rng.choice(BOARD_SIZE * BOARD_SIZE, size=len(values), replace=False)
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        for flat_pos, value in zip(positions.tolist(), values):
            board[flat_pos // BOARD_SIZE, flat_pos % BOARD_SIZE] = value

        max_tile = board_max_tile(board)
        state = SimState(
            board=board,
            preview=preview,
            small_counts=small_counts,
            small_pos=small_pos,
            small_seen_total=0,
            span_small_pos=0,
            large_pending=False,
            max_tile=max_tile,
            move_count=0,
            game_over=max_tile >= TERMINAL_TILE,
        )
        return state

    def legal_actions(self, state: SimState) -> list[int]:
        if state.game_over or int(np.max(state.board, initial=0)) >= TERMINAL_TILE:
            return []
        legal: list[int] = []
        for action in range(4):
            _after, eligible = simulate_base_move(state.board, action)
            if eligible:
                legal.append(action)
        return legal

    def legal_mask(self, state: SimState) -> np.ndarray:
        mask = np.zeros(4, dtype=bool)
        for action in self.legal_actions(state):
            mask[action] = True
        return mask

    def step(self, state: SimState, action: Direction) -> tuple[SimState, StepInfo]:
        if state.game_over:
            return state, StepInfo(False, None, None, 0, 0, [])

        before_score = score_board(state.board)
        shifted, eligible_positions = simulate_base_move(state.board, action)
        if not eligible_positions:
            return state, StepInfo(False, None, None, 0, 0, [])

        merge_delta = score_board(shifted) - before_score
        terminal_merge = bool(np.any(shifted == TERMINAL_TILE))
        if terminal_merge:
            next_state = self._replace_state(
                state,
                board=shifted,
                max_tile=TERMINAL_TILE,
                move_count=state.move_count + 1,
                game_over=True,
            )
            return next_state, StepInfo(
                moved=True,
                inserted_value=None,
                inserted_pos=None,
                merge_score_delta=merge_delta,
                score_delta=score_board(shifted) - before_score,
                eligible_positions=eligible_positions,
                terminal_merge=True,
            )

        pos_idx = int(self.rng.integers(len(eligible_positions)))
        inserted_pos = eligible_positions[pos_idx]
        inserted_value = self._sample_insert_value(state.preview)
        board_after = shifted.copy()
        board_after[inserted_pos] = inserted_value

        next_state = self._state_after_insert(state, board_after)
        info = StepInfo(
            moved=True,
            inserted_value=inserted_value,
            inserted_pos=inserted_pos,
            merge_score_delta=merge_delta,
            score_delta=score_board(board_after) - before_score,
            eligible_positions=eligible_positions,
            terminal_merge=False,
        )
        return next_state, info

    def transition_outcomes(self, state: SimState, action: Direction, include_info: bool = True) -> list[tuple[float, SimState, StepInfo]]:
        if state.game_over:
            return []
        before_score = score_board(state.board) if include_info else 0
        shifted, eligible_positions = simulate_base_move(state.board, action)
        if not eligible_positions:
            return []

        merge_delta = score_board(shifted) - before_score if include_info else 0
        if bool(np.any(shifted == TERMINAL_TILE)):
            next_state = self._replace_state(
                state,
                board=shifted,
                max_tile=TERMINAL_TILE,
                move_count=state.move_count + 1,
                game_over=True,
            )
            score_delta = score_board(shifted) - before_score if include_info else 0
            info = StepInfo(True, None, None, merge_delta, score_delta, eligible_positions, True)
            return [(1.0, next_state, info)]

        insert_options = self._insert_value_options(state.preview)
        slot_prob = 1.0 / len(eligible_positions)
        outcomes: list[tuple[float, SimState, StepInfo]] = []
        for pos in eligible_positions:
            for inserted_value, value_prob in insert_options:
                board_after = shifted.copy()
                board_after[pos] = inserted_value
                consumed = self._consume_preview(
                    state.small_counts,
                    state.small_pos,
                    state.small_seen_total,
                    state.span_small_pos,
                    state.large_pending,
                    state.preview.label,
                )
                max_tile = board_max_tile(board_after)
                preview_options = self.preview_options(
                    consumed[0],
                    consumed[1],
                    consumed[2],
                    consumed[3],
                    consumed[4],
                    max_tile,
                )
                score_delta = score_board(board_after) - before_score if include_info else 0
                for preview_option in preview_options:
                    next_state = SimState(
                        board=board_after.copy(),
                        preview=preview_option.preview,
                        small_counts=consumed[0].copy(),
                        small_pos=consumed[1],
                        small_seen_total=consumed[2],
                        span_small_pos=consumed[3],
                        large_pending=consumed[4],
                        max_tile=max_tile,
                        move_count=state.move_count + 1,
                        game_over=False,
                    )
                    info = StepInfo(
                        True,
                        inserted_value,
                        pos,
                        merge_delta,
                        score_delta,
                        eligible_positions,
                        False,
                    )
                    outcomes.append((slot_prob * value_prob * preview_option.probability, next_state, info))
        return outcomes

    def tile_cycle_snapshot(self, state: SimState) -> tuple[dict[str, int], int, int, int, bool, int]:
        return (
            state.small_counts.copy(),
            int(state.small_pos),
            int(state.small_seen_total),
            int(state.span_small_pos),
            bool(state.large_pending),
            int(state.max_tile),
        )

    def state_from_snapshot(
        self,
        board: Sequence[Sequence[int]] | np.ndarray,
        preview: Preview,
        snapshot: tuple[dict[str, int], int, int, int, bool, int],
        move_count: int = 0,
    ) -> SimState:
        counts, small_pos, small_seen_total, span_small_pos, large_pending, max_tile = snapshot
        state = SimState(
            board=np.asarray(board, dtype=np.int32).copy(),
            preview=preview,
            small_counts={str(k): int(v) for k, v in counts.items()},
            small_pos=int(small_pos),
            small_seen_total=int(small_seen_total),
            span_small_pos=int(span_small_pos),
            large_pending=bool(large_pending),
            max_tile=int(max_tile),
            move_count=int(move_count),
            game_over=False,
        )
        state.max_tile = max(state.max_tile, board_max_tile(state.board))
        state.game_over = state.max_tile >= TERMINAL_TILE or not self.legal_actions(state)
        return state

    def preview_options(
        self,
        small_counts: dict[str, int],
        small_pos: int,
        small_seen_total: int,
        span_small_pos: int,
        large_pending: bool,
        max_tile: int,
    ) -> list[PreviewOption]:
        large_prob = self._large_probability(small_seen_total, span_small_pos, large_pending, max_tile)
        windows = self.bonus_windows(max_tile)
        if not windows:
            large_prob = 0.0

        small_slots_left = max(1, SMALL_BAG_SIZE - int(small_pos))
        small_scale = max(0.0, 1.0 - large_prob)
        options: list[PreviewOption] = []
        for label in SMALL_LABELS:
            remaining = max(0, int(small_counts.get(label, 0)))
            prob = (remaining / small_slots_left) * small_scale
            if prob > 0.0:
                options.append(PreviewOption(preview_from_label(label), prob))

        if large_prob > 0.0:
            per_window = large_prob / len(windows)
            for window in windows:
                options.append(PreviewOption(preview_from_label("large_candidates", window), per_window))

        total = sum(option.probability for option in options)
        if not options or total <= 0.0:
            raise RuntimeError(f"No preview options for counts={small_counts} pos={small_pos} max={max_tile}")
        if abs(total - 1.0) > 1e-9:
            options = [PreviewOption(option.preview, option.probability / total) for option in options]
        return options

    def bonus_values(self, max_tile: int) -> list[int]:
        if int(max_tile) < BONUS_TRIGGER_TILE:
            return []
        limit = max(6, int(max_tile) // 4)
        values: list[int] = []
        value = 6
        while value <= limit:
            values.append(value)
            value *= 2
        return values

    def bonus_windows(self, max_tile: int) -> list[tuple[int, int, int]]:
        values = self.bonus_values(max_tile)
        if len(values) < 3:
            return []
        return [tuple(values[idx : idx + 3]) for idx in range(len(values) - 2)]

    def safe_smalls_until_large_possible(self, state: SimState) -> Optional[int]:
        if state.max_tile < BONUS_TRIGGER_TILE:
            return None
        if state.small_seen_total < LARGE_DELAY_PREVIEWS:
            return max(0, LARGE_DELAY_PREVIEWS - state.small_seen_total)
        if state.large_pending:
            return 0
        return max(0, LARGE_SPAN_SMALLS - state.span_small_pos)

    def _sample_preview(
        self,
        small_counts: dict[str, int],
        small_pos: int,
        small_seen_total: int,
        span_small_pos: int,
        large_pending: bool,
        max_tile: int,
    ) -> Preview:
        options = self.preview_options(small_counts, small_pos, small_seen_total, span_small_pos, large_pending, max_tile)
        probs = np.asarray([option.probability for option in options], dtype=np.float64)
        idx = int(self.rng.choice(len(options), p=probs))
        return options[idx].preview

    def _sample_small_label(self, small_counts: dict[str, int], small_pos: int) -> str:
        slots = max(1, SMALL_BAG_SIZE - int(small_pos))
        draw = int(self.rng.integers(slots))
        cumulative = 0
        for label in SMALL_LABELS:
            cumulative += max(0, int(small_counts[label]))
            if draw < cumulative:
                return label
        for label in reversed(SMALL_LABELS):
            if small_counts[label] > 0:
                return label
        raise RuntimeError(f"No small labels left in counts={small_counts}")

    def _sample_insert_value(self, preview: Preview) -> int:
        if preview.kind != "bonus":
            if preview.value is None:
                raise RuntimeError(f"Small preview missing value: {preview}")
            return int(preview.value)
        if not preview.candidates:
            raise RuntimeError(f"Bonus preview missing candidates: {preview}")
        idx = int(self.rng.integers(len(preview.candidates)))
        return int(preview.candidates[idx])

    def _insert_value_options(self, preview: Preview) -> list[tuple[int, float]]:
        if preview.kind != "bonus":
            if preview.value is None:
                raise RuntimeError(f"Small preview missing value: {preview}")
            return [(int(preview.value), 1.0)]
        if not preview.candidates:
            raise RuntimeError(f"Bonus preview missing candidates: {preview}")
        prob = 1.0 / len(preview.candidates)
        return [(int(value), prob) for value in preview.candidates]

    def _state_after_insert(self, state: SimState, board_after: np.ndarray) -> SimState:
        counts, small_pos, small_seen_total, span_small_pos, large_pending = self._consume_preview(
            state.small_counts,
            state.small_pos,
            state.small_seen_total,
            state.span_small_pos,
            state.large_pending,
            state.preview.label,
        )
        max_tile = board_max_tile(board_after)
        preview = self._sample_preview(counts, small_pos, small_seen_total, span_small_pos, large_pending, max_tile)
        next_state = SimState(
            board=board_after,
            preview=preview,
            small_counts=counts,
            small_pos=small_pos,
            small_seen_total=small_seen_total,
            span_small_pos=span_small_pos,
            large_pending=large_pending,
            max_tile=max_tile,
            move_count=state.move_count + 1,
            game_over=False,
        )
        next_state.game_over = bool(np.all(board_after != 0)) and not self.legal_actions(next_state)
        return next_state

    def _consume_preview(
        self,
        small_counts: dict[str, int],
        small_pos: int,
        small_seen_total: int,
        span_small_pos: int,
        large_pending: bool,
        label: str,
    ) -> tuple[dict[str, int], int, int, int, bool]:
        counts = {key: int(small_counts.get(key, 0)) for key in SMALL_LABELS}
        small_pos = int(small_pos)
        small_seen_total = int(small_seen_total)
        span_small_pos = int(span_small_pos)
        large_pending = bool(large_pending)

        is_large = label == "large_candidates"
        if not is_large:
            small_pos += 1
            small_seen_total += 1
            if small_seen_total == LARGE_DELAY_PREVIEWS:
                span_small_pos = 0
                large_pending = True
            elif small_seen_total > LARGE_DELAY_PREVIEWS:
                span_small_pos += 1

        if label in counts and counts[label] > 0:
            counts[label] -= 1

        if is_large and large_pending:
            large_pending = False

        if small_pos >= SMALL_BAG_SIZE:
            counts = {"red": 4, "blue": 4, "gray": 4}
            small_pos = 0

        if small_seen_total >= LARGE_DELAY_PREVIEWS:
            span_small_pos = min(span_small_pos, LARGE_SPAN_SMALLS)
            if not large_pending and span_small_pos >= LARGE_SPAN_SMALLS:
                span_small_pos = 0
                large_pending = True

        return counts, small_pos, small_seen_total, span_small_pos, large_pending

    def _large_probability(self, small_seen_total: int, span_small_pos: int, large_pending: bool, max_tile: int) -> float:
        if int(max_tile) < BONUS_TRIGGER_TILE:
            return 0.0
        if int(small_seen_total) < LARGE_DELAY_PREVIEWS:
            return 0.0
        if not large_pending:
            return 0.0
        remaining_slots = max(1, LARGE_SPAN_SMALLS + 1 - int(span_small_pos))
        return 1.0 / remaining_slots

    def _replace_state(self, state: SimState, **kwargs: object) -> SimState:
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
        values.update(kwargs)
        if not isinstance(values["board"], np.ndarray):
            values["board"] = np.asarray(values["board"], dtype=np.int32)
        return SimState(**values)  # type: ignore[arg-type]

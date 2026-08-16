"""N-tuple afterstate value function for Threes.

The value model is NumPy-only so it can be used by the simulator, search, and
future live-hint code without pulling in torch.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from threes_rl.sim import SCORE_BY_VALUE, SimState, ThreesSim, rank_for_value, score_board, simulate_base_move

NUM_RANKS = 16
RANK_BY_VALUE = {int(value): min(NUM_RANKS - 1, rank_for_value(int(value))) for value in SCORE_BY_VALUE}
PHASE4_NAMES = ("early_lt384", "mid_384_768", "late_1536", "endgame_3072p")
CORNER_RISK_NAMES = ("low_corner_risk", "medium_corner_risk", "high_corner_risk")


@dataclass(frozen=True)
class Symmetry:
    name: str
    cell_perm: tuple[int, ...]
    action_perm: tuple[int, ...]


def _coord_to_index(row: int, col: int) -> int:
    return row * 4 + col


def _index_to_coord(index: int) -> tuple[int, int]:
    return divmod(index, 4)


def _symmetry_specs():
    return (
        ("identity", lambda r, c: (r, c)),
        ("rot90", lambda r, c: (c, 3 - r)),
        ("rot180", lambda r, c: (3 - r, 3 - c)),
        ("rot270", lambda r, c: (3 - c, r)),
        ("flip_h", lambda r, c: (r, 3 - c)),
        ("flip_v", lambda r, c: (3 - r, c)),
        ("transpose", lambda r, c: (c, r)),
        ("anti_transpose", lambda r, c: (3 - c, 3 - r)),
    )


def _build_symmetries() -> tuple[Symmetry, ...]:
    direction_vectors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    vector_to_action = {vec: idx for idx, vec in enumerate(direction_vectors)}
    out: list[Symmetry] = []
    for name, transform in _symmetry_specs():
        perm = [0] * 16
        for old_idx in range(16):
            old_r, old_c = _index_to_coord(old_idx)
            new_r, new_c = transform(old_r, old_c)
            perm[_coord_to_index(new_r, new_c)] = old_idx

        action_perm: list[int] = []
        for dr, dc in direction_vectors:
            # Pick an interior-ish origin that remains in-bounds for this step.
            r = 1 if dr <= 0 else 2
            c = 1 if dc <= 0 else 2
            r0, c0 = transform(r, c)
            r1, c1 = transform(r + dr, c + dc)
            action_perm.append(vector_to_action[(r1 - r0, c1 - c0)])
        out.append(Symmetry(name, tuple(perm), tuple(action_perm)))
    return tuple(out)


SYMMETRIES = _build_symmetries()
SYMMETRY_CELL_PERMS = tuple(np.asarray(symmetry.cell_perm, dtype=np.intp) for symmetry in SYMMETRIES)


def rank_board(board: np.ndarray) -> np.ndarray:
    arr = np.asarray(board, dtype=np.int32).reshape(-1)
    return np.asarray(
        [RANK_BY_VALUE.get(int(value), min(NUM_RANKS - 1, rank_for_value(int(value)))) for value in arr],
        dtype=np.int16,
    )


def _rows() -> list[tuple[int, ...]]:
    return [tuple(range(row * 4, row * 4 + 4)) for row in range(4)]


def _cols() -> list[tuple[int, ...]]:
    return [tuple(range(col, 16, 4)) for col in range(4)]


def _squares_2x2() -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for row in range(3):
        for col in range(3):
            patterns.append(
                (
                    _coord_to_index(row, col),
                    _coord_to_index(row, col + 1),
                    _coord_to_index(row + 1, col),
                    _coord_to_index(row + 1, col + 1),
                )
            )
    return patterns


def _edge_rectangles_2x3() -> list[tuple[int, ...]]:
    return [
        (0, 1, 2, 4, 5, 6),
        (2, 3, 6, 7, 10, 11),
        (9, 10, 11, 13, 14, 15),
        (4, 5, 8, 9, 12, 13),
    ]


def _big_six_tuples() -> list[tuple[int, ...]]:
    return [
        # Dense local rectangles.
        (0, 1, 2, 4, 5, 6),
        (1, 2, 3, 5, 6, 7),
        (0, 1, 4, 5, 8, 9),
        (1, 2, 5, 6, 9, 10),
        # Edge and corner paths that can represent anchored snake-like shapes.
        (0, 1, 2, 3, 4, 5),
        (0, 4, 8, 12, 1, 5),
        (0, 1, 4, 5, 6, 7),
        (0, 4, 5, 6, 8, 12),
    ]


def patterns_for_set(name: str) -> list[tuple[int, ...]]:
    if name == "tiny":
        return [(0, 1, 4, 5), (0, 4, 8, 12)]
    if name == "small":
        return _rows() + _cols() + _squares_2x2()
    if name == "default":
        return _rows() + _cols() + _squares_2x2() + _edge_rectangles_2x3()
    if name == "big6":
        return _big_six_tuples()
    raise ValueError(f"Unsupported n-tuple pattern set: {name}")


def max_tile_excluding_free_starter(board: np.ndarray, starter_tile: int | None) -> int:
    arr = np.asarray(board, dtype=np.int32).copy()
    if starter_tile is None:
        return int(arr.max(initial=0))
    matches = np.argwhere(arr == int(starter_tile))
    if len(matches):
        match_idx = 0
        for idx, (row, col) in enumerate(matches):
            if int(row) == 0 and int(col) == 0:
                match_idx = idx
                break
        row, col = matches[match_idx]
        arr[int(row), int(col)] = 0
    return int(arr.max(initial=0))


def phase4_index_for_board(board: np.ndarray, starter_tile: int | None = 1536) -> int:
    built_max = max_tile_excluding_free_starter(board, starter_tile)
    if built_max < 384:
        return 0
    if built_max < 1536:
        return 1
    if built_max < 3072:
        return 2
    return 3


def corner_risk_bucket_for_board(board: np.ndarray, starter_tile: int | None = 1536) -> str:
    arr = np.asarray(board, dtype=np.int32)
    top_left = int(arr[0, 0])
    board_max = int(arr.max(initial=0))
    empty_count = int(np.count_nonzero(arr == 0))
    built_max = max_tile_excluding_free_starter(arr, starter_tile)
    risk = 0
    if built_max >= 384 and top_left != board_max:
        risk += 2
    if empty_count <= 2:
        risk += 2
    elif empty_count <= 4:
        risk += 1
    if starter_tile is not None and top_left not in (0, int(starter_tile), board_max):
        risk += 1
    if risk >= 3:
        return "high_corner_risk"
    if risk >= 1:
        return "medium_corner_risk"
    return "low_corner_risk"


def corner_risk_index_for_board(board: np.ndarray, starter_tile: int | None = 1536) -> int:
    return CORNER_RISK_NAMES.index(corner_risk_bucket_for_board(board, starter_tile=starter_tile))


def stage_names_for_mode(stage_mode: str) -> list[str]:
    if stage_mode == "phase4":
        return list(PHASE4_NAMES)
    if stage_mode == "phase4_corner3":
        return [f"{phase}/{risk}" for phase in PHASE4_NAMES for risk in CORNER_RISK_NAMES]
    raise ValueError(f"Unsupported staged n-tuple mode: {stage_mode}")


def stage_index_for_board(board: np.ndarray, stage_mode: str, starter_tile: int | None = 1536) -> int:
    phase_idx = phase4_index_for_board(board, starter_tile=starter_tile)
    if stage_mode == "phase4":
        return phase_idx
    if stage_mode == "phase4_corner3":
        return phase_idx * len(CORNER_RISK_NAMES) + corner_risk_index_for_board(board, starter_tile=starter_tile)
    raise ValueError(f"Unsupported staged n-tuple mode: {stage_mode}")


def index_for_pattern(ranks: np.ndarray, pattern: tuple[int, ...]) -> int:
    index = 0
    for cell in pattern:
        index = index * NUM_RANKS + int(ranks[cell])
    return int(index)


class NtupleValue:
    def __init__(
        self,
        patterns: Iterable[tuple[int, ...]],
        *,
        init: float = 0.0,
        tables: list[np.ndarray] | None = None,
        tc_sum_tables: list[np.ndarray] | None = None,
        tc_abs_tables: list[np.ndarray] | None = None,
        pattern_set: str = "custom",
    ) -> None:
        self.patterns = [tuple(int(cell) for cell in pattern) for pattern in patterns]
        if not self.patterns:
            raise ValueError("At least one pattern is required")
        self.pattern_set = pattern_set
        if tables is None:
            self.tables = [
                np.full(NUM_RANKS ** len(pattern), float(init), dtype=np.float32)
                for pattern in self.patterns
            ]
        else:
            if len(tables) != len(self.patterns):
                raise ValueError(f"Expected {len(self.patterns)} tables, got {len(tables)}")
            self.tables = [np.asarray(table, dtype=np.float32) for table in tables]
        self.tc_sum_tables = self._coerce_tc_tables(tc_sum_tables, "tc_sum_tables")
        self.tc_abs_tables = self._coerce_tc_tables(tc_abs_tables, "tc_abs_tables")
        self.feature_count = len(self.patterns) * len(SYMMETRIES)

    def _coerce_tc_tables(self, tables: list[np.ndarray] | None, name: str) -> list[np.ndarray] | None:
        if tables is None:
            return None
        if len(tables) != len(self.patterns):
            raise ValueError(f"Expected {len(self.patterns)} {name}, got {len(tables)}")
        return [np.asarray(table, dtype=np.float32) for table in tables]

    def enable_temporal_coherence(self) -> None:
        if self.tc_sum_tables is None:
            self.tc_sum_tables = [np.zeros_like(table, dtype=np.float32) for table in self.tables]
        if self.tc_abs_tables is None:
            self.tc_abs_tables = [np.zeros_like(table, dtype=np.float32) for table in self.tables]

    @classmethod
    def from_pattern_set(cls, name: str, *, init: float = 0.0) -> "NtupleValue":
        return cls(patterns_for_set(name), init=init, pattern_set=name)

    def clone(self) -> "NtupleValue":
        return NtupleValue(
            self.patterns,
            tables=[np.array(table, dtype=np.float32, copy=True) for table in self.tables],
            tc_sum_tables=None
            if self.tc_sum_tables is None
            else [np.array(table, dtype=np.float32, copy=True) for table in self.tc_sum_tables],
            tc_abs_tables=None
            if self.tc_abs_tables is None
            else [np.array(table, dtype=np.float32, copy=True) for table in self.tc_abs_tables],
            pattern_set=self.pattern_set,
        )

    def indices(self, board: np.ndarray) -> list[tuple[int, int]]:
        base_ranks = rank_board(board)
        out: list[tuple[int, int]] = []
        for cell_perm in SYMMETRY_CELL_PERMS:
            ranks = base_ranks[cell_perm]
            for table_idx, pattern in enumerate(self.patterns):
                out.append((table_idx, index_for_pattern(ranks, pattern)))
        return out

    def value(self, board: np.ndarray) -> float:
        base_ranks = rank_board(board)
        total = 0.0
        for cell_perm in SYMMETRY_CELL_PERMS:
            ranks = base_ranks[cell_perm]
            for table_idx, pattern in enumerate(self.patterns):
                total += float(self.tables[table_idx][index_for_pattern(ranks, pattern)])
        return total

    def update(self, board: np.ndarray, target: float, alpha: float) -> float:
        current = self.value(board)
        delta = float(target) - current
        step = float(alpha) * delta / float(self.feature_count)
        for table_idx, index in self.indices(board):
            self.tables[table_idx][index] += step
        return delta

    def update_tc(self, board: np.ndarray, target: float, alpha: float) -> float:
        self.enable_temporal_coherence()
        if self.tc_sum_tables is None or self.tc_abs_tables is None:
            raise RuntimeError("Temporal-coherence tables were not initialized")
        current = self.value(board)
        delta = float(target) - current
        abs_delta = abs(delta)
        for table_idx, index in self.indices(board):
            self.tc_sum_tables[table_idx][index] += delta
            self.tc_abs_tables[table_idx][index] += abs_delta
            tc_abs = float(self.tc_abs_tables[table_idx][index])
            tc_sum = float(self.tc_sum_tables[table_idx][index])
            scale = 1.0 if tc_abs <= 1e-12 else abs(tc_sum) / tc_abs
            self.tables[table_idx][index] += float(alpha) * scale * delta / float(self.feature_count)
        return delta

    def save(self, path: Path, extra_meta: dict[str, object] | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "pattern_set": self.pattern_set,
            "patterns": [list(pattern) for pattern in self.patterns],
            "num_ranks": NUM_RANKS,
            "tables": [f"table_{idx:03d}.npy" for idx in range(len(self.tables))],
        }
        if self.tc_sum_tables is not None:
            meta["tc_sum_tables"] = [f"tc_sum_{idx:03d}.npy" for idx in range(len(self.tc_sum_tables))]
        if self.tc_abs_tables is not None:
            meta["tc_abs_tables"] = [f"tc_abs_{idx:03d}.npy" for idx in range(len(self.tc_abs_tables))]
        if extra_meta:
            meta.update(extra_meta)
        for idx, table in enumerate(self.tables):
            np.save(path / f"table_{idx:03d}.npy", table)
        if self.tc_sum_tables is not None:
            for idx, table in enumerate(self.tc_sum_tables):
                np.save(path / f"tc_sum_{idx:03d}.npy", table)
        if self.tc_abs_tables is not None:
            for idx, table in enumerate(self.tc_abs_tables):
                np.save(path / f"tc_abs_{idx:03d}.npy", table)
        (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path, *, mmap_mode: str | None = None) -> "NtupleValue":
        meta = json.loads((path / "meta.json").read_text())
        if meta.get("value_type") == "residual_staged_composite":
            return ResidualStagedNtupleValue.load(path, meta=meta, mmap_mode=mmap_mode)  # type: ignore[return-value]
        if meta.get("value_type") == "staged":
            return StagedNtupleValue.load(path, meta=meta, mmap_mode=mmap_mode)  # type: ignore[return-value]
        patterns = [tuple(int(cell) for cell in pattern) for pattern in meta["patterns"]]
        tables = [np.load(path / table_name, mmap_mode=mmap_mode) for table_name in meta["tables"]]
        tc_sum_tables = [np.load(path / table_name, mmap_mode=mmap_mode) for table_name in meta.get("tc_sum_tables", [])]
        tc_abs_tables = [np.load(path / table_name, mmap_mode=mmap_mode) for table_name in meta.get("tc_abs_tables", [])]
        return cls(
            patterns,
            tables=tables,
            tc_sum_tables=tc_sum_tables or None,
            tc_abs_tables=tc_abs_tables or None,
            pattern_set=str(meta.get("pattern_set", "custom")),
        )


class StagedNtupleValue:
    """Board-phase-conditioned collection of afterstate n-tuple values."""

    def __init__(
        self,
        stages: list[NtupleValue | None],
        *,
        stage_mode: str = "phase4",
        starter_tile: int | None = 1536,
        stage_names: list[str] | None = None,
        pattern_set: str | None = None,
        lazy_init: float = 0.0,
        promotion_enabled: bool = False,
        promotion_copy_tc: bool = True,
        promotion_masks: list[list[np.ndarray] | None] | None = None,
        touched_masks: list[list[np.ndarray] | None] | None = None,
        promotion_counts: list[int] | None = None,
        feature_access_counts: list[int] | None = None,
    ) -> None:
        expected_stage_names = stage_names_for_mode(stage_mode)
        if len(stages) != len(expected_stage_names):
            raise ValueError(f"{stage_mode} requires {len(expected_stage_names)} stages, got {len(stages)}")
        self.stages = list(stages)
        self.stage_mode = stage_mode
        self.starter_tile = starter_tile
        self.stage_names = list(stage_names) if stage_names is not None else expected_stage_names
        first_stage = next((stage for stage in self.stages if stage is not None), None)
        if pattern_set is None and first_stage is None:
            raise ValueError("pattern_set is required when all staged tables are lazy")
        self.pattern_set = pattern_set or first_stage.pattern_set  # type: ignore[union-attr]
        self.lazy_init = float(lazy_init)
        self.promotion_enabled = bool(promotion_enabled)
        self.promotion_copy_tc = bool(promotion_copy_tc)
        if self.promotion_enabled and not self.promotion_copy_tc:
            raise ValueError("Stage promotion must copy both weights and temporal-coherence state")
        self.feature_count = (
            first_stage.feature_count
            if first_stage is not None
            else len(patterns_for_set(self.pattern_set)) * len(SYMMETRIES)
        )
        self._patterns = list(first_stage.patterns) if first_stage is not None else patterns_for_set(self.pattern_set)
        self._validate_stage_patterns()
        self.promotion_masks = self._coerce_masks(promotion_masks, promoted=True)
        self.touched_masks = self._coerce_masks(touched_masks, promoted=False)
        stage_count = len(self.stages)
        self.promotion_counts = list(promotion_counts) if promotion_counts is not None else [0] * stage_count
        self.feature_access_counts = (
            list(feature_access_counts) if feature_access_counts is not None else [0] * stage_count
        )
        if len(self.promotion_counts) != stage_count or len(self.feature_access_counts) != stage_count:
            raise ValueError("Stage metric counts must match the number of stages")

    def _validate_stage_patterns(self) -> None:
        expected = self._patterns
        for idx, stage in enumerate(self.stages):
            if stage is None:
                continue
            if stage.patterns != expected:
                raise ValueError(
                    f"Incompatible patterns for stage promotion at stage {idx}: "
                    f"expected {expected}, got {stage.patterns}"
                )

    def _empty_masks(self) -> list[np.ndarray]:
        return [np.zeros(NUM_RANKS ** len(pattern), dtype=bool) for pattern in self._patterns]

    def _coerce_masks(
        self,
        masks: list[list[np.ndarray] | None] | None,
        *,
        promoted: bool,
    ) -> list[list[np.ndarray] | None]:
        if not self.promotion_enabled:
            return [None for _stage in self.stages]
        if masks is None:
            return [None if promoted and idx == 0 else self._empty_masks() for idx in range(len(self.stages))]
        if len(masks) != len(self.stages):
            raise ValueError("Stage masks must match the number of stages")
        out: list[list[np.ndarray] | None] = []
        for stage_idx, stage_masks in enumerate(masks):
            if promoted and stage_idx == 0:
                out.append(None)
                continue
            if stage_masks is None or len(stage_masks) != len(self._patterns):
                raise ValueError(f"Missing or incompatible masks for stage {stage_idx}")
            coerced = [np.asarray(mask, dtype=bool) for mask in stage_masks]
            for pattern, mask in zip(self._patterns, coerced):
                expected_size = NUM_RANKS ** len(pattern)
                if mask.size != expected_size:
                    raise ValueError(
                        f"Incompatible mask size for stage {stage_idx}: expected {expected_size}, got {mask.size}"
                    )
            out.append(coerced)
        return out

    @classmethod
    def from_pattern_set(
        cls,
        name: str,
        *,
        init: float = 0.0,
        stage_mode: str = "phase4",
        starter_tile: int | None = 1536,
        lazy: bool = False,
        promotion_enabled: bool = False,
    ) -> "StagedNtupleValue":
        if promotion_enabled:
            raise ValueError("Stage weight promotion requires a parent model; use from_base_model")
        stage_names = stage_names_for_mode(stage_mode)
        stages: list[NtupleValue | None]
        if lazy:
            stages = [None for _stage in stage_names]
        else:
            stages = [NtupleValue.from_pattern_set(name, init=init) for _stage in stage_names]
        return cls(
            stages,
            stage_mode=stage_mode,
            starter_tile=starter_tile,
            pattern_set=name,
            lazy_init=init,
        )

    @classmethod
    def from_base_model(
        cls,
        base_model: NtupleValue,
        *,
        stage_mode: str = "phase4",
        starter_tile: int | None = 1536,
        promotion_enabled: bool = False,
        promotion_copy_tc: bool = True,
    ) -> "StagedNtupleValue":
        stage_names = stage_names_for_mode(stage_mode)
        if promotion_enabled:
            stages: list[NtupleValue | None] = [base_model.clone()]
            stages.extend(
                NtupleValue(base_model.patterns, init=0.0, pattern_set=base_model.pattern_set)
                for _stage in stage_names[1:]
            )
        else:
            stages = [base_model.clone() for _stage in stage_names]
        return cls(
            stages,
            stage_mode=stage_mode,
            starter_tile=starter_tile,
            pattern_set=base_model.pattern_set,
            promotion_enabled=promotion_enabled,
            promotion_copy_tc=promotion_copy_tc,
        )

    def stage_index(self, board: np.ndarray) -> int:
        return stage_index_for_board(board, self.stage_mode, self.starter_tile)

    def stage_name(self, board: np.ndarray) -> str:
        return self.stage_names[self.stage_index(board)]

    def _new_stage(self) -> NtupleValue:
        return NtupleValue(self._patterns, init=self.lazy_init, pattern_set=self.pattern_set)

    def stage_for_board(self, board: np.ndarray, *, create: bool = False) -> NtupleValue | None:
        stage_idx = self.stage_index(board)
        stage = self.stages[stage_idx]
        if stage is None and create:
            stage = self._new_stage()
            self.stages[stage_idx] = stage
        return stage

    def enable_temporal_coherence(self) -> None:
        for stage in self.stages:
            if stage is not None:
                stage.enable_temporal_coherence()

    def _effective_entry(self, stage_idx: int, table_idx: int, index: int, field: str) -> float:
        stage = self.stages[stage_idx]
        if stage_idx == 0 or not self.promotion_enabled:
            if stage is None:
                return 0.0
            table_list = getattr(stage, field)
            if table_list is None:
                return 0.0
            return float(table_list[table_idx][index])
        masks = self.promotion_masks[stage_idx]
        if stage is not None and masks is not None and bool(masks[table_idx][index]):
            table_list = getattr(stage, field)
            if table_list is None:
                return 0.0
            return float(table_list[table_idx][index])
        return self._effective_entry(stage_idx - 1, table_idx, index, field)

    def _promote_indices(self, stage_idx: int, indices: list[tuple[int, int]]) -> None:
        if not self.promotion_enabled or stage_idx == 0:
            return
        stage = self.stages[stage_idx]
        if stage is None:
            stage = self._new_stage()
            self.stages[stage_idx] = stage
        masks = self.promotion_masks[stage_idx]
        if masks is None:
            raise RuntimeError(f"Promotion masks missing for stage {stage_idx}")
        for table_idx, index in set(indices):
            if bool(masks[table_idx][index]):
                continue
            stage.tables[table_idx][index] = self._effective_entry(stage_idx - 1, table_idx, index, "tables")
            if self.promotion_copy_tc:
                if stage.tc_sum_tables is not None:
                    stage.tc_sum_tables[table_idx][index] = self._effective_entry(
                        stage_idx - 1, table_idx, index, "tc_sum_tables"
                    )
                if stage.tc_abs_tables is not None:
                    stage.tc_abs_tables[table_idx][index] = self._effective_entry(
                        stage_idx - 1, table_idx, index, "tc_abs_tables"
                    )
            masks[table_idx][index] = True
            self.promotion_counts[stage_idx] += 1

    def _mark_touched(self, stage_idx: int, indices: list[tuple[int, int]]) -> None:
        self.feature_access_counts[stage_idx] += len(indices)
        masks = self.touched_masks[stage_idx]
        if masks is None:
            return
        for table_idx, index in set(indices):
            masks[table_idx][index] = True

    def value(self, board: np.ndarray) -> float:
        stage_idx = self.stage_index(board)
        stage = self.stages[stage_idx]
        if not self.promotion_enabled:
            return 0.0 if stage is None else stage.value(board)
        indices = (stage if stage is not None else self.stages[0]).indices(board)  # type: ignore[union-attr]
        return float(sum(self._effective_entry(stage_idx, table_idx, index, "tables") for table_idx, index in indices))

    def update(self, board: np.ndarray, target: float, alpha: float) -> float:
        stage = self.stage_for_board(board, create=True)
        if stage is None:
            raise RuntimeError("staged value table could not be initialized")
        if not self.promotion_enabled:
            return stage.update(board, target, alpha)
        stage_idx = self.stage_index(board)
        indices = stage.indices(board)
        self._promote_indices(stage_idx, indices)
        self._mark_touched(stage_idx, indices)
        return stage.update(board, target, alpha)

    def update_tc(self, board: np.ndarray, target: float, alpha: float) -> float:
        stage = self.stage_for_board(board, create=True)
        if stage is None:
            raise RuntimeError("staged value table could not be initialized")
        if not self.promotion_enabled:
            return stage.update_tc(board, target, alpha)
        self.enable_temporal_coherence()
        stage_idx = self.stage_index(board)
        indices = stage.indices(board)
        self._promote_indices(stage_idx, indices)
        self._mark_touched(stage_idx, indices)
        return stage.update_tc(board, target, alpha)

    def stage_metrics(self) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        total_bytes = 0
        capacity = sum(NUM_RANKS ** len(pattern) for pattern in self._patterns)
        for idx, stage in enumerate(self.stages):
            table_bytes = 0
            if stage is not None:
                arrays = list(stage.tables)
                arrays.extend(stage.tc_sum_tables or [])
                arrays.extend(stage.tc_abs_tables or [])
                table_bytes = sum(int(array.nbytes) for array in arrays)
            mask_bytes = sum(int(mask.nbytes) for mask in (self.promotion_masks[idx] or []))
            touched_bytes = sum(int(mask.nbytes) for mask in (self.touched_masks[idx] or []))
            unique_touched = sum(int(np.count_nonzero(mask)) for mask in (self.touched_masks[idx] or []))
            promoted = int(self.promotion_counts[idx])
            memory_bytes = table_bytes + mask_bytes + touched_bytes
            total_bytes += memory_bytes
            rows.append(
                {
                    "index": idx,
                    "name": self.stage_names[idx],
                    "table_entries_touched": unique_touched,
                    "entries_promoted": promoted,
                    "unique_features": unique_touched,
                    "feature_accesses": int(self.feature_access_counts[idx]),
                    "promotion_fraction": float(promoted / capacity) if capacity else 0.0,
                    "table_capacity": int(capacity),
                    "memory_bytes": int(memory_bytes),
                }
            )
        return {
            "promotion_enabled": self.promotion_enabled,
            "promotion_semantics": "copy_weight_and_tc_on_first_training_access"
            if self.promotion_enabled
            else "disabled",
            "stages": rows,
            "memory_bytes": int(total_bytes),
        }

    @staticmethod
    def _save_masks(path: Path, prefix: str, masks: list[list[np.ndarray] | None]) -> list[list[dict[str, object]] | None]:
        manifest: list[list[dict[str, object]] | None] = []
        for stage_idx, stage_masks in enumerate(masks):
            if stage_masks is None:
                manifest.append(None)
                continue
            stage_manifest: list[dict[str, object]] = []
            for table_idx, mask in enumerate(stage_masks):
                name = f"{prefix}_stage_{stage_idx:02d}_table_{table_idx:03d}.npy"
                np.save(path / name, np.packbits(mask, bitorder="little"))
                stage_manifest.append({"file": name, "size": int(mask.size)})
            manifest.append(stage_manifest)
        return manifest

    @staticmethod
    def _load_masks(path: Path, manifest: object) -> list[list[np.ndarray] | None]:
        if not isinstance(manifest, list):
            raise ValueError("Invalid stage mask manifest")
        masks: list[list[np.ndarray] | None] = []
        for stage_manifest in manifest:
            if stage_manifest is None:
                masks.append(None)
                continue
            if not isinstance(stage_manifest, list):
                raise ValueError("Invalid per-stage mask manifest")
            stage_masks: list[np.ndarray] = []
            for row in stage_manifest:
                if not isinstance(row, dict):
                    raise ValueError("Invalid mask entry")
                packed = np.load(path / str(row["file"]))
                stage_masks.append(np.unpackbits(packed, bitorder="little", count=int(row["size"])).astype(bool))
            masks.append(stage_masks)
        return masks

    def save(self, path: Path, extra_meta: dict[str, object] | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        stage_dirs: list[str | None] = []
        for idx, stage in enumerate(self.stages):
            if stage is None:
                stage_dirs.append(None)
                continue
            safe_stage_name = self.stage_names[idx].replace("/", "_").replace("=", "_")
            stage_dir = f"stage_{idx:02d}_{safe_stage_name}"
            stage.save(path / stage_dir)
            stage_dirs.append(stage_dir)
        meta = {
            "value_type": "staged",
            "stage_mode": self.stage_mode,
            "starter_tile": self.starter_tile,
            "stage_names": self.stage_names,
            "stage_dirs": stage_dirs,
            "pattern_set": self.pattern_set,
            "lazy_init": self.lazy_init,
            "promotion_enabled": self.promotion_enabled,
            "promotion_copy_tc": self.promotion_copy_tc,
            "promotion_semantics": "copy_weight_and_tc_on_first_training_access"
            if self.promotion_enabled
            else "disabled",
            "promotion_counts": self.promotion_counts,
            "feature_access_counts": self.feature_access_counts,
            "patterns": [list(pattern) for pattern in self._patterns],
        }
        if self.promotion_enabled:
            meta["promotion_masks"] = self._save_masks(path, "promoted", self.promotion_masks)
            meta["touched_masks"] = self._save_masks(path, "touched", self.touched_masks)
        if extra_meta:
            meta.update(extra_meta)
        (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        meta: dict[str, object] | None = None,
        mmap_mode: str | None = None,
    ) -> "StagedNtupleValue":
        loaded_meta = meta if meta is not None else json.loads((path / "meta.json").read_text())
        stage_dirs = loaded_meta["stage_dirs"]  # type: ignore[index]
        stages = [
            None if stage_dir is None else NtupleValue.load(path / str(stage_dir), mmap_mode=mmap_mode)
            for stage_dir in stage_dirs
        ]
        starter_tile = loaded_meta.get("starter_tile", 1536)
        first_stage = next((stage for stage in stages if stage is not None), None)
        pattern_set = loaded_meta.get("pattern_set")
        if pattern_set is None:
            if first_stage is None:
                raise ValueError("Lazy staged checkpoint is missing pattern_set metadata")
            pattern_set = first_stage.pattern_set
        promotion_enabled = bool(loaded_meta.get("promotion_enabled", False))
        promotion_masks = (
            cls._load_masks(path, loaded_meta.get("promotion_masks")) if promotion_enabled else None
        )
        touched_masks = cls._load_masks(path, loaded_meta.get("touched_masks")) if promotion_enabled else None
        loaded = cls(
            stages,
            stage_mode=str(loaded_meta.get("stage_mode", "phase4")),
            starter_tile=None if starter_tile is None else int(starter_tile),
            stage_names=[str(name) for name in loaded_meta.get("stage_names", stage_names_for_mode(str(loaded_meta.get("stage_mode", "phase4"))))],  # type: ignore[arg-type]
            pattern_set=str(pattern_set),
            lazy_init=float(loaded_meta.get("lazy_init", 0.0)),
            promotion_enabled=promotion_enabled,
            promotion_copy_tc=bool(loaded_meta.get("promotion_copy_tc", True)),
            promotion_masks=promotion_masks,
            touched_masks=touched_masks,
            promotion_counts=[int(value) for value in loaded_meta.get("promotion_counts", [0] * len(stages))],  # type: ignore[arg-type]
            feature_access_counts=[int(value) for value in loaded_meta.get("feature_access_counts", [0] * len(stages))],  # type: ignore[arg-type]
        )
        expected_patterns = loaded_meta.get("patterns")
        if expected_patterns is not None:
            normalized = [tuple(int(cell) for cell in pattern) for pattern in expected_patterns]  # type: ignore[union-attr]
            if normalized != loaded._patterns:
                raise ValueError("Incompatible patterns in staged promotion checkpoint")
        return loaded


class ResidualStagedNtupleValue:
    """Frozen blended leaf plus a trainable promoted stage residual."""

    VALUE_TYPE = "residual_staged_composite"

    def __init__(
        self,
        *,
        frozen_policy_spec: str,
        base_checkpoint: Path,
        blend_specs: list[tuple[Path, float]],
        phase_blend_specs: list[tuple[Path, float, int | str]],
        residual: StagedNtupleValue,
    ) -> None:
        self.frozen_policy_spec = str(frozen_policy_spec)
        self.base_checkpoint = Path(base_checkpoint)
        self.blend_specs = [(Path(path), float(weight)) for path, weight in blend_specs]
        self.phase_blend_specs = [
            (Path(path), float(weight), int(gate) if isinstance(gate, int) else str(gate))
            for path, weight, gate in phase_blend_specs
        ]
        self.residual = residual
        self.pattern_set = residual.pattern_set
        self.feature_count = residual.feature_count
        self.stage_mode = residual.stage_mode
        self.stage_names = residual.stage_names
        self.starter_tile = residual.starter_tile
        if self.stage_mode != "phase4":
            raise ValueError("Residual composite currently requires phase4 staging")
        if not residual.promotion_enabled:
            raise ValueError("Residual composite requires per-feature stage promotion")

        total_blend_weight = sum(weight for _path, weight in self.blend_specs)
        if not 0.0 <= total_blend_weight <= 1.0:
            raise ValueError("Frozen blend weights must sum to the interval [0, 1]")
        for _path, weight in self.blend_specs:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Frozen blend weight must be in [0, 1], got {weight}")
        for _path, weight, gate in self.phase_blend_specs:
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Frozen phase blend weight must be in [0, 1], got {weight}")
            self._validate_gate(gate)
        self.base_weight = 1.0 - total_blend_weight

        self.base_model = NtupleValue.load(self.base_checkpoint, mmap_mode="r")
        self.blend_models = [
            (path, NtupleValue.load(path, mmap_mode="r"), weight)
            for path, weight in self.blend_specs
        ]
        self.phase_blend_models = [
            (path, NtupleValue.load(path, mmap_mode="r"), weight, gate)
            for path, weight, gate in self.phase_blend_specs
        ]
        self._validate_frozen_patterns()

    @staticmethod
    def _validate_gate(gate: int | str) -> None:
        if isinstance(gate, int) and 0 <= gate < len(PHASE4_NAMES):
            return
        if gate == "all":
            return
        raise ValueError(f"Unsupported residual frozen-leaf gate: {gate}")

    def _gate_active(self, board: np.ndarray, gate: int | str) -> bool:
        if gate == "all":
            return True
        if isinstance(gate, int):
            return phase4_index_for_board(board, starter_tile=self.starter_tile) >= gate
        raise ValueError(f"Unsupported residual frozen-leaf gate: {gate}")

    @property
    def frozen_models(self) -> list[NtupleValue]:
        return [
            self.base_model,
            *(model for _path, model, _weight in self.blend_models),
            *(model for _path, model, _weight, _gate in self.phase_blend_models),
        ]

    @property
    def frozen_checkpoints(self) -> list[Path]:
        return [
            self.base_checkpoint,
            *(path for path, _weight in self.blend_specs),
            *(path for path, _weight, _gate in self.phase_blend_specs),
        ]

    def frozen_source_fingerprint(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for checkpoint in self.frozen_checkpoints:
            files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
            signature = hashlib.sha256()
            total_bytes = 0
            for path in files:
                stat = path.stat()
                relative = str(path.relative_to(checkpoint))
                total_bytes += int(stat.st_size)
                signature.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
            rows.append(
                {
                    "checkpoint": str(checkpoint),
                    "files": len(files),
                    "bytes": total_bytes,
                    "stat_sha256": signature.hexdigest(),
                }
            )
        return rows

    @property
    def frozen_arrays(self) -> list[np.ndarray]:
        arrays: list[np.ndarray] = []
        for model in self.frozen_models:
            if isinstance(model, StagedNtupleValue):
                for stage in model.stages:
                    if stage is None:
                        continue
                    arrays.extend(stage.tables)
                    arrays.extend(stage.tc_sum_tables or [])
                    arrays.extend(stage.tc_abs_tables or [])
            else:
                arrays.extend(model.tables)
                arrays.extend(model.tc_sum_tables or [])
                arrays.extend(model.tc_abs_tables or [])
        return arrays

    def _validate_frozen_patterns(self) -> None:
        expected = self.residual._patterns
        for model in self.frozen_models:
            model_patterns = getattr(model, "patterns", getattr(model, "_patterns", None))
            if model.pattern_set != self.pattern_set or list(model_patterns or []) != list(expected):
                raise ValueError("Frozen incumbent and residual pattern sets are incompatible")

    @classmethod
    def from_frozen_blend(
        cls,
        *,
        frozen_policy_spec: str,
        base_checkpoint: Path,
        blend_specs: list[tuple[Path, float]],
        phase_blend_specs: list[tuple[Path, float, int | str]],
        pattern_set: str = "default",
        starter_tile: int | None = 1536,
    ) -> "ResidualStagedNtupleValue":
        stages = [NtupleValue.from_pattern_set(pattern_set, init=0.0) for _stage in PHASE4_NAMES]
        residual = StagedNtupleValue(
            stages,
            stage_mode="phase4",
            starter_tile=starter_tile,
            pattern_set=pattern_set,
            promotion_enabled=True,
            promotion_copy_tc=True,
        )
        return cls(
            frozen_policy_spec=frozen_policy_spec,
            base_checkpoint=base_checkpoint,
            blend_specs=blend_specs,
            phase_blend_specs=phase_blend_specs,
            residual=residual,
        )

    def frozen_value(self, board: np.ndarray) -> float:
        base_value = float(self.base_model.value(board))
        active_phase_models = [
            (model, weight)
            for _path, model, weight, gate in self.phase_blend_models
            if self._gate_active(board, gate)
        ]
        active_phase_weight = sum(weight for _model, weight in active_phase_models)
        value = (self.base_weight - active_phase_weight) * base_value
        for _path, model, weight in self.blend_models:
            if weight > 0.0:
                value += weight * float(model.value(board))
        for model, weight in active_phase_models:
            if weight > 0.0:
                value += weight * float(model.value(board))
        return float(value)

    def residual_value(self, board: np.ndarray) -> float:
        return float(self.residual.value(board))

    def value(self, board: np.ndarray) -> float:
        return float(self.frozen_value(board) + self.residual_value(board))

    def update(self, board: np.ndarray, target: float, alpha: float) -> float:
        residual_target = float(target) - self.frozen_value(board)
        return float(self.residual.update(board, residual_target, alpha))

    def update_tc(self, board: np.ndarray, target: float, alpha: float) -> float:
        residual_target = float(target) - self.frozen_value(board)
        return float(self.residual.update_tc(board, residual_target, alpha))

    def enable_temporal_coherence(self) -> None:
        self.residual.enable_temporal_coherence()

    def stage_metrics(self) -> dict[str, object]:
        payload = self.residual.stage_metrics()
        return {
            **payload,
            "value_type": self.VALUE_TYPE,
            "frozen_component_count": len(self.frozen_models),
            "residual_only_updates": True,
        }

    def save(self, path: Path, extra_meta: dict[str, object] | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        residual_dir = "residual"
        self.residual.save(path / residual_dir)
        meta: dict[str, object] = {
            "value_type": self.VALUE_TYPE,
            "pattern_set": self.pattern_set,
            "stage_mode": self.stage_mode,
            "stage_names": self.stage_names,
            "starter_tile": self.starter_tile,
            "promotion_enabled": True,
            "promotion_copy_tc": True,
            "promotion_semantics": "copy_weight_and_tc_on_first_training_access_residual_only",
            "frozen_policy_spec": self.frozen_policy_spec,
            "base_checkpoint": str(self.base_checkpoint),
            "blend_specs": [
                {"checkpoint": str(path_value), "weight": weight}
                for path_value, weight in self.blend_specs
            ],
            "phase_blend_specs": [
                {"checkpoint": str(path_value), "weight": weight, "gate": gate}
                for path_value, weight, gate in self.phase_blend_specs
            ],
            "residual_dir": residual_dir,
            "residual_only_updates": True,
        }
        if extra_meta:
            meta.update(extra_meta)
        (path / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        meta: dict[str, object] | None = None,
        mmap_mode: str | None = None,
    ) -> "ResidualStagedNtupleValue":
        loaded_meta = meta if meta is not None else json.loads((path / "meta.json").read_text())
        residual = StagedNtupleValue.load(path / str(loaded_meta["residual_dir"]), mmap_mode=mmap_mode)
        blend_specs = [
            (Path(str(row["checkpoint"])), float(row["weight"]))
            for row in loaded_meta.get("blend_specs", [])  # type: ignore[union-attr]
        ]
        phase_blend_specs = [
            (
                Path(str(row["checkpoint"])),
                float(row["weight"]),
                int(row["gate"]) if isinstance(row["gate"], int) else str(row["gate"]),
            )
            for row in loaded_meta.get("phase_blend_specs", [])  # type: ignore[union-attr]
        ]
        return cls(
            frozen_policy_spec=str(loaded_meta["frozen_policy_spec"]),
            base_checkpoint=Path(str(loaded_meta["base_checkpoint"])),
            blend_specs=blend_specs,
            phase_blend_specs=phase_blend_specs,
            residual=residual,
        )


def post_spawn_state_value(value_model: NtupleValue, state: SimState, sim: ThreesSim) -> float:
    """Greedy value of a state after a spawn has occurred.

    The n-tuple table is an afterstate value: it scores the shifted board before
    the next random insertion. Once a tile has spawned, the next decision value
    is the best immediate merge score plus the value of that next afterstate.
    """

    best: float | None = None
    before_score = score_board(state.board)
    for action in sim.legal_actions(state):
        shifted, eligible = simulate_base_move(state.board, action)
        if not eligible:
            continue
        merge_delta = score_board(shifted) - before_score
        value = float(merge_delta + value_model.value(shifted))
        if best is None or value > best:
            best = value
    return 0.0 if best is None else float(best)


def expected_afterstate_target(value_model: NtupleValue, state: SimState, sim: ThreesSim, action: int) -> tuple[float, np.ndarray | None]:
    """Return the Bellman target for the afterstate reached by ``action``.

    The returned target excludes the merge score that created the afterstate and
    includes the expected spawn score plus the greedy value of the post-spawn
    state. Terminal 12288 afterstates have target 0 because no spawn follows.
    """

    shifted, eligible = simulate_base_move(state.board, action)
    if not eligible:
        return -1e30, None

    before_score = score_board(state.board)
    merge_delta = score_board(shifted) - before_score
    if np.any(shifted == 12288):
        return 0.0, shifted

    expected = 0.0
    for probability, next_state, info in sim.transition_outcomes(state, action, include_next_preview=False):
        spawn_score = float(info.score_delta - merge_delta)
        expected += float(probability) * (spawn_score + post_spawn_state_value(value_model, next_state, sim))
    return float(expected), shifted


def afterstate_action_value(value_model: NtupleValue, state: SimState, sim: ThreesSim, action: int) -> tuple[float, np.ndarray | None]:
    shifted, eligible = simulate_base_move(state.board, action)
    if not eligible:
        return -1e30, None
    immediate = score_board(shifted) - score_board(state.board)
    if np.any(shifted == 12288):
        return float(immediate), shifted
    target, afterstate = expected_afterstate_target(value_model, state, sim, action)
    return float(immediate + target), afterstate


def choose_action(
    value_model: NtupleValue,
    state: SimState,
    sim: ThreesSim,
    rng: np.random.Generator,
    *,
    epsilon: float = 0.0,
) -> tuple[int, np.ndarray]:
    legal = sim.legal_actions(state)
    if not legal:
        return 0, state.board.copy()
    if epsilon > 0.0 and float(rng.random()) < epsilon:
        action = int(legal[int(rng.integers(len(legal)))])
        shifted, _eligible = simulate_base_move(state.board, action)
        return action, shifted
    best_value: float | None = None
    best: list[tuple[int, np.ndarray]] = []
    for action in legal:
        value, afterstate = afterstate_action_value(value_model, state, sim, action)
        if afterstate is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best = [(action, afterstate)]
        elif value == best_value:
            best.append((action, afterstate))
    if not best:
        return 0, state.board.copy()
    return best[int(rng.integers(len(best)))]


class NtuplePolicy:
    def __init__(self, checkpoint: Path, epsilon: float = 0.0) -> None:
        self.value_model = NtupleValue.load(checkpoint, mmap_mode="r")
        self.epsilon = float(epsilon)
        self.name = f"ntuple:{checkpoint}"

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        action, _afterstate = choose_action(self.value_model, state, sim, rng, epsilon=self.epsilon)
        return int(action)

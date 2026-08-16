"""Separate exact native leaf/transition kernel for the frozen K1 assay."""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from threes_rl.c1_search_optimization import (
    BatchedPersistentPolicy,
    VectorizedCompositeLeaf,
)
from threes_rl.expectimax import NtupleExpectimaxPolicy


VERSION = "k1_compiled_exact_kernel_v1"
ABI_VERSION = 1
SOURCE_PATH = Path("threes_rl/k1_exact_kernel.c")
WRAPPER_PATH = Path("threes_rl/k1_compiled_kernel.py")
COMPILER = Path(
    "/Applications/Xcode.app/Contents/Developer/Toolchains/"
    "XcodeDefault.xctoolchain/usr/bin/clang"
)
COMPILER_SHA256 = (
    "7def90dd8829726686213a747fc5bff1583df933dae5edc55d755479e0bfe00a"
)
SDK_ROOT = Path(
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
    "Developer/SDKs/MacOSX26.5.sdk"
)
COMPILE_FLAGS = (
    "-O3",
    "-std=c11",
    "-fPIC",
    "-dynamiclib",
    "-isysroot",
    str(SDK_ROOT),
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-associative-math",
    "-Wall",
    "-Wextra",
    "-Werror",
)
MODEL_COUNT = 4
PHASE_COUNT = 4
PATTERN_COUNT = 21
SYMMETRY_COUNT = 8
MAX_PATTERN_LENGTH = 6
PHASE_COEFFICIENTS = np.asarray(
    (
        (0.75, 0.25, 0.00, 0.00),
        (0.70, 0.25, 0.05, 0.00),
        (0.70, 0.25, 0.05, 0.00),
        (0.60, 0.25, 0.05, 0.10),
    ),
    dtype=np.float64,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_command(output_path: Path) -> list[str]:
    return [
        str(COMPILER),
        *COMPILE_FLAGS,
        "-o",
        str(output_path),
        str(SOURCE_PATH),
    ]


def build_native_kernel(output_path: Path) -> dict[str, Any]:
    if sha256_path(COMPILER) != COMPILER_SHA256:
        raise ValueError("K1 compiler binary changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(output_path)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "K1 native build failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    if not output_path.is_file():
        raise RuntimeError("K1 native build did not create the library")
    return {
        "version": VERSION,
        "compiler": str(COMPILER),
        "compiler_sha256": sha256_path(COMPILER),
        "source": str(SOURCE_PATH),
        "source_sha256": sha256_path(SOURCE_PATH),
        "wrapper": str(WRAPPER_PATH),
        "wrapper_sha256": sha256_path(WRAPPER_PATH),
        "flags": list(COMPILE_FLAGS),
        "command": command,
        "library": str(output_path),
        "library_sha256": sha256_path(output_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


class NativeKernel:
    """Loaded exact K1 native library and frozen incumbent table bindings."""

    def __init__(
        self,
        library_path: Path,
        policy: NtupleExpectimaxPolicy,
    ) -> None:
        self.library_path = Path(library_path)
        self.library_sha256 = sha256_path(self.library_path)
        self.library = ctypes.CDLL(str(self.library_path.resolve()))
        self._bind_functions()
        if int(self.library.k1_kernel_abi_version()) != ABI_VERSION:
            raise ValueError("K1 native ABI mismatch")
        self.eval_call_count = 0
        self.base_move_call_count = 0
        self.score_call_count = 0
        self.post_spawn_call_count = 0
        self.reference_leaf = VectorizedCompositeLeaf(policy)
        self._configure_leaf(policy)

    def _bind_functions(self) -> None:
        i32p = ctypes.POINTER(ctypes.c_int32)
        i16p = ctypes.POINTER(ctypes.c_int16)
        i8p = ctypes.POINTER(ctypes.c_int8)
        i64p = ctypes.POINTER(ctypes.c_int64)
        f32pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        f64p = ctypes.POINTER(ctypes.c_double)
        self.library.k1_kernel_abi_version.argtypes = []
        self.library.k1_kernel_abi_version.restype = ctypes.c_int
        self.library.k1_base_move.argtypes = [
            i32p,
            ctypes.c_int,
            i32p,
            i32p,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.library.k1_base_move.restype = ctypes.c_int
        self.library.k1_score_board.argtypes = [
            i32p,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.library.k1_score_board.restype = ctypes.c_int
        self.library.k1_eval_composite.argtypes = [
            i32p,
            ctypes.c_size_t,
            i16p,
            ctypes.c_size_t,
            i8p,
            i8p,
            f32pp,
            i64p,
            f64p,
            f64p,
        ]
        self.library.k1_eval_composite.restype = ctypes.c_int
        self.library.k1_post_spawn_rows.argtypes = [
            i32p,
            i16p,
            ctypes.c_size_t,
            i8p,
            i8p,
            f32pp,
            i64p,
            f64p,
            i32p,
            i32p,
            i32p,
            i32p,
            i64p,
            i64p,
            f64p,
        ]
        self.library.k1_post_spawn_rows.restype = ctypes.c_int

    @staticmethod
    def _model_tables(model: Any, phase: int) -> Sequence[np.ndarray]:
        if hasattr(model, "tables"):
            return model.tables
        stages = getattr(model, "stages", None)
        if stages is None or len(stages) != PHASE_COUNT:
            raise ValueError("K1 requires exact phase4 or unstaged models")
        stage = stages[phase]
        if stage is None:
            raise ValueError("K1 frozen incumbent has a missing phase stage")
        return stage.tables

    def _configure_leaf(self, policy: NtupleExpectimaxPolicy) -> None:
        if (
            policy.ensemble_mode != "blend"
            or policy.geometry_weight != 0.0
            or len(policy.blend_models) != 1
            or len(policy.phase_blend_models) != 2
            or policy.bonus_models
        ):
            raise ValueError("K1 policy is not the frozen incumbent blend")
        models = (
            policy.value_model,
            policy.blend_models[0][1],
            policy.phase_blend_models[0][1],
            policy.phase_blend_models[1][1],
        )
        if len(self.reference_leaf.patterns) != PATTERN_COUNT:
            raise ValueError("K1 pattern count mismatch")
        self.pattern_lengths = np.asarray(
            [len(pattern) for pattern in self.reference_leaf.patterns],
            dtype=np.int8,
        )
        self.pattern_cells = np.full(
            (PATTERN_COUNT, SYMMETRY_COUNT, MAX_PATTERN_LENGTH),
            -1,
            dtype=np.int8,
        )
        for pattern_index, cells in enumerate(self.reference_leaf.cells):
            self.pattern_cells[
                pattern_index, :, : cells.shape[1]
            ] = np.asarray(cells, dtype=np.int8)
        self.rank_lut = np.ascontiguousarray(
            self.reference_leaf.rank_lut,
            dtype=np.int16,
        )
        pointer_type = ctypes.POINTER(ctypes.c_float)
        pointers: list[Any] = []
        lengths: list[int] = []
        self.table_arrays: list[np.ndarray] = []
        for model in models:
            patterns = tuple(
                getattr(model, "patterns", getattr(model, "_patterns", ()))
            )
            if patterns != tuple(self.reference_leaf.patterns):
                raise ValueError("K1 component pattern mismatch")
            for phase in range(PHASE_COUNT):
                tables = self._model_tables(model, phase)
                if len(tables) != PATTERN_COUNT:
                    raise ValueError("K1 component table count mismatch")
                for table in tables:
                    array = np.asarray(table)
                    if (
                        array.dtype != np.float32
                        or array.ndim != 1
                        or not array.flags.c_contiguous
                    ):
                        raise ValueError(
                            "K1 table must be contiguous one-dimensional float32"
                        )
                    self.table_arrays.append(array)
                    pointers.append(
                        array.ctypes.data_as(pointer_type)
                    )
                    lengths.append(int(array.size))
        pointer_array_type = pointer_type * len(pointers)
        self.table_pointers = pointer_array_type(*pointers)
        self.table_lengths = np.asarray(lengths, dtype=np.int64)
        self.phase_coefficients = np.ascontiguousarray(
            PHASE_COEFFICIENTS,
            dtype=np.float64,
        )
        self.binding_manifest = {
            "version": VERSION,
            "library_sha256": self.library_sha256,
            "pattern_count": PATTERN_COUNT,
            "pattern_lengths": self.pattern_lengths.astype(int).tolist(),
            "pattern_cells_sha256": hashlib.sha256(
                self.pattern_cells.tobytes()
            ).hexdigest(),
            "rank_lut_sha256": hashlib.sha256(
                self.rank_lut.tobytes()
            ).hexdigest(),
            "phase_coefficients": self.phase_coefficients.tolist(),
            "phase_coefficients_sha256": hashlib.sha256(
                self.phase_coefficients.tobytes()
            ).hexdigest(),
            "table_count": len(self.table_arrays),
            "table_payload_sha256": hashlib.sha256(
                "|".join(
                    hashlib.sha256(table.view(np.uint8)).hexdigest()
                    for table in self.table_arrays
                ).encode()
            ).hexdigest(),
            "table_lengths_sha256": hashlib.sha256(
                self.table_lengths.tobytes()
            ).hexdigest(),
        }
        self.binding_manifest["binding_sha256"] = canonical_json_hash(
            self.binding_manifest
        )

    @staticmethod
    def _i32_pointer(array: np.ndarray) -> Any:
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def evaluate_many(self, boards: Sequence[np.ndarray]) -> np.ndarray:
        if not boards:
            return np.empty(0, dtype=np.float64)
        self.eval_call_count += 1
        board_array = np.ascontiguousarray(boards, dtype=np.int32).reshape(-1, 16)
        output = np.empty(len(board_array), dtype=np.float64)
        code = self.library.k1_eval_composite(
            self._i32_pointer(board_array),
            len(board_array),
            self.rank_lut.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int16)
            ),
            self.rank_lut.size,
            self.pattern_lengths.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int8)
            ),
            self.pattern_cells.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int8)
            ),
            self.table_pointers,
            self.table_lengths.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int64)
            ),
            self.phase_coefficients.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if code != 0:
            raise ValueError(f"K1 native leaf rejected input with code {code}")
        return output

    def base_move(
        self,
        board: np.ndarray,
        action: int,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        self.base_move_call_count += 1
        input_board = np.ascontiguousarray(board, dtype=np.int32).reshape(16)
        output_board = np.empty(16, dtype=np.int32)
        eligible = np.empty(8, dtype=np.int32)
        eligible_count = ctypes.c_int32()
        code = self.library.k1_base_move(
            self._i32_pointer(input_board),
            int(action),
            self._i32_pointer(output_board),
            self._i32_pointer(eligible),
            ctypes.byref(eligible_count),
        )
        if code != 0:
            raise ValueError(f"K1 native base move rejected input with code {code}")
        count = int(eligible_count.value)
        return (
            output_board.reshape(4, 4),
            tuple(
                (int(eligible[2 * index]), int(eligible[2 * index + 1]))
                for index in range(count)
            ),
        )

    def score_board(self, board: np.ndarray) -> int:
        self.score_call_count += 1
        input_board = np.ascontiguousarray(board, dtype=np.int32).reshape(16)
        output = ctypes.c_int64()
        code = self.library.k1_score_board(
            self._i32_pointer(input_board),
            ctypes.byref(output),
        )
        if code != 0:
            raise ValueError(f"K1 native score rejected input with code {code}")
        return int(output.value)

    def post_spawn_rows(
        self,
        board: np.ndarray,
    ) -> tuple[
        int,
        list[
            tuple[
                int,
                np.ndarray,
                tuple[tuple[int, int], ...],
                int,
                float,
            ]
        ],
    ]:
        self.post_spawn_call_count += 1
        input_board = np.ascontiguousarray(board, dtype=np.int32).reshape(16)
        boards = np.empty((4, 16), dtype=np.int32)
        eligible = np.empty((4, 8), dtype=np.int32)
        eligible_counts = np.empty(4, dtype=np.int32)
        legal_flags = np.empty(4, dtype=np.int32)
        before_score = np.empty(1, dtype=np.int64)
        after_scores = np.empty(4, dtype=np.int64)
        leaf_values = np.empty(4, dtype=np.float64)
        code = self.library.k1_post_spawn_rows(
            self._i32_pointer(input_board),
            self.rank_lut.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int16)
            ),
            self.rank_lut.size,
            self.pattern_lengths.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int8)
            ),
            self.pattern_cells.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int8)
            ),
            self.table_pointers,
            self.table_lengths.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int64)
            ),
            self.phase_coefficients.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            self._i32_pointer(boards),
            self._i32_pointer(eligible),
            self._i32_pointer(eligible_counts),
            self._i32_pointer(legal_flags),
            before_score.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int64)
            ),
            after_scores.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int64)
            ),
            leaf_values.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
        )
        if code != 0:
            raise ValueError(
                f"K1 native post-spawn rows rejected input with code {code}"
            )
        rows = []
        for action in range(4):
            if not bool(legal_flags[action]):
                continue
            count = int(eligible_counts[action])
            rows.append(
                (
                    action,
                    boards[action].reshape(4, 4).copy(),
                    tuple(
                        (
                            int(eligible[action, 2 * index]),
                            int(eligible[action, 2 * index + 1]),
                        )
                        for index in range(count)
                    ),
                    int(after_scores[action]),
                    float(leaf_values[action]),
                )
            )
        return int(before_score[0]), rows


class NativeCompositeLeaf:
    def __init__(self, kernel: NativeKernel) -> None:
        self.kernel = kernel

    def evaluate_many(self, boards: list[np.ndarray]) -> np.ndarray:
        return self.kernel.evaluate_many(boards)


class K1CompiledPolicy(BatchedPersistentPolicy):
    """C1-exact depth-3 search with only leaf/base-move/score in native C."""

    def __init__(
        self,
        *args: Any,
        library_path: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.native_kernel = NativeKernel(library_path, self)
        self.vectorized_leaf = NativeCompositeLeaf(self.native_kernel)

    def _score_board(self, board: np.ndarray) -> int:
        key = self._fast_board_key(board)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        value = self.native_kernel.score_board(board)
        self._score_cache[key] = value
        return value

    def _base_move(
        self,
        board: np.ndarray,
        action: int,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        key = (self._fast_board_key(board), int(action))
        cached = self._base_move_cache.get(key)
        if cached is not None:
            return cached
        value = self.native_kernel.base_move(board, int(action))
        self._base_move_cache[key] = value
        return value

    def _afterstate_value(self, board: np.ndarray) -> float:
        key = self._fast_board_key(board)
        cached = self._afterstate_cache.get(key)
        if cached is not None:
            return cached
        value = float(self.vectorized_leaf.evaluate_many([board])[0])
        self._afterstate_cache[key] = value
        return value

    def _post_spawn_state_value(self, state: Any, sim: Any) -> float:
        if state.game_over:
            return 0.0
        key = self._fast_board_key(state.board)
        cached = self._post_spawn_cache.get(key)
        if cached is not None:
            return cached
        before_score, rows = self.native_kernel.post_spawn_rows(state.board)
        self._score_cache[key] = before_score
        legal = []
        best: float | None = None
        for action, shifted, eligible, after_score, leaf_value in rows:
            legal.append(action)
            shifted_key = self._fast_board_key(shifted)
            self._base_move_cache[(key, action)] = (shifted, eligible)
            self._score_cache[shifted_key] = after_score
            self._afterstate_cache[shifted_key] = leaf_value
            value = float(after_score - before_score + leaf_value)
            if best is None or value > best:
                best = value
        self._legal_cache[key] = tuple(legal)
        value = 0.0 if best is None else float(best)
        self._post_spawn_cache[key] = value
        return value

    def adaptive_values(self, state: Any, sim: Any) -> dict[str, Any]:
        before = (
            self.native_kernel.eval_call_count,
            self.native_kernel.base_move_call_count,
            self.native_kernel.score_call_count,
            self.native_kernel.post_spawn_call_count,
        )
        result = super().adaptive_values(state, sim)
        result["compiled_calls"] = {
            "leaf": self.native_kernel.eval_call_count - before[0],
            "base_move": self.native_kernel.base_move_call_count - before[1],
            "score": self.native_kernel.score_call_count - before[2],
            "post_spawn": self.native_kernel.post_spawn_call_count - before[3],
        }
        return result


def clone_k1(
    base: NtupleExpectimaxPolicy,
    library_path: Path,
) -> K1CompiledPolicy:
    from threes_rl.r2a_adaptive_expectimax import CHANCE_LIMIT, NODE_BUDGET

    return K1CompiledPolicy(
        base.checkpoint,
        depth=3,
        chance_limit=CHANCE_LIMIT,
        blend_specs=list(base.blend_specs),
        phase_blend_specs=list(base.phase_blend_specs),
        bonus_specs=list(base.bonus_specs),
        tie_margin=base.tie_margin,
        tie_breaker=base.tie_breaker,
        ensemble_mode=base.ensemble_mode,
        geometry_weight=base.geometry_weight,
        geometry_min_tile=base.geometry_min_tile,
        node_budget=NODE_BUDGET,
        library_path=library_path,
    )

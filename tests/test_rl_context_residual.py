from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.context_residual import (
    ContextResidualModel,
    INPUT_WIDTH,
    OUTPUT_NAMES,
    context_metadata,
    encode_state,
)
from threes_rl.sim import ThreesSim, preview_from_label


def context_state(sim: ThreesSim, preview: str):
    board = np.asarray(
        [
            [1536, 3, 6, 12],
            [1, 2, 3, 6],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    return sim.state_from_snapshot(
        board,
        preview_from_label(preview),
        ({"red": 3, "blue": 2, "gray": 2}, 5, 120, 14, True, 1536),
        move_count=140,
    )


class ContextResidualTests(unittest.TestCase):
    def test_zero_residual_is_exact_identity_and_equal_capacity(self) -> None:
        sim = ThreesSim.from_stream_ids(deck_stream_id=1, slot_stream_id=2, starter_tile=1536)
        state = context_state(sim, "red")
        board_only = ContextResidualModel(mode="board_stage_only")
        with_context = ContextResidualModel(mode="board_plus_context")

        self.assertEqual(board_only.parameter_count, with_context.parameter_count)
        np.testing.assert_array_equal(board_only.w1, with_context.w1)
        self.assertEqual(board_only.residual_value(state, sim), 0.0)
        self.assertEqual(with_context.residual_value(state, sim), 0.0)
        self.assertEqual(with_context.total_value(12345.5, state, sim), 12345.5)

    def test_board_only_mask_is_context_invariant_but_context_mode_can_distinguish(self) -> None:
        sim = ThreesSim.from_stream_ids(deck_stream_id=3, slot_stream_id=4, starter_tile=1536)
        red = context_state(sim, "red")
        blue = context_state(sim, "blue")
        board_model = ContextResidualModel(mode="board_stage_only")
        context_model = ContextResidualModel(mode="board_plus_context")
        for model in (board_model, context_model):
            model.w1.fill(0.0)
            model.w2.fill(0.0)
            model.w1[32, 0] = 1.0
            model.w1[33, 0] = -1.0
            model.w2[0, OUTPUT_NAMES.index("expected_return_residual")] = 1.0

        self.assertEqual(board_model.residual_value(red, sim), board_model.residual_value(blue, sim))
        self.assertNotEqual(context_model.residual_value(red, sim), context_model.residual_value(blue, sim))
        self.assertEqual(encode_state(red, sim, mode="board_stage_only").shape, (INPUT_WIDTH,))

    def test_save_load_round_trip_and_schema_guard(self) -> None:
        sim = ThreesSim.from_stream_ids(deck_stream_id=5, slot_stream_id=6, starter_tile=1536)
        state = context_state(sim, "gray")
        model = ContextResidualModel(mode="board_plus_context")
        model.w2[0, 0] = 0.25
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            path = Path(tmp) / "model"
            model.save(path)
            loaded = ContextResidualModel.load(path)

            self.assertEqual(loaded.mode, model.mode)
            self.assertEqual(loaded.parameter_count, model.parameter_count)
            self.assertEqual(loaded.predict(state, sim), model.predict(state, sim))

            meta_path = path / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["schema_sha256"] = "incompatible"
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                ContextResidualModel.load(path)

    def test_context_metadata_has_normalized_next_distributions_without_mutation(self) -> None:
        sim = ThreesSim.from_stream_ids(deck_stream_id=7, slot_stream_id=8, starter_tile=1536)
        state = context_state(sim, "red")
        board_before = state.board.copy()
        counts_before = dict(state.small_counts)

        metadata = context_metadata(state, sim)

        total_next = sum(metadata["next_small_joint"].values()) + sum(
            metadata["next_plus_value_joint"].values()
        )
        self.assertAlmostEqual(total_next, 1.0)
        if metadata["p_plus_next"] > 0:
            self.assertAlmostEqual(sum(metadata["next_plus_value_conditional"].values()), 1.0)
        np.testing.assert_array_equal(state.board, board_before)
        self.assertEqual(state.small_counts, counts_before)


if __name__ == "__main__":
    unittest.main()

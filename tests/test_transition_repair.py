import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import state_hunt as sh
import window_stream as ws


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class TransitionRepairTests(unittest.TestCase):
    def _gray_cells_from_board_fixture(self, name: str):
        board_arr = np.array(Image.open(FIXTURE_DIR / name).convert("RGB"))
        gray_boxes, _grid_meta = ws._board_cell_boxes(board_arr, inset_ratio=ws.GRAY_INSET_RATIO)
        lookup = {}
        for row, col, _outer_box, inner_box in gray_boxes:
            x0, y0, x1, y1 = inner_box
            lookup[(row, col)] = board_arr[y0:y1, x0:x1]
        return lookup

    def test_failure_cell_keeps_96_as_supported_candidate(self):
        gray_cells = self._gray_cells_from_board_fixture("move69_after_board.png")
        scores = ws.gray_label_scores(gray_cells[(0, 1)])
        self.assertIn("96", scores)
        self.assertGreaterEqual(scores["96"], ws.gray_label_score_threshold())

    def test_unique_gray_repair_recovers_move69(self):
        blue = ws.SMALL_COLOR_MAP["blue"]
        red = ws.SMALL_COLOR_MAP["red"]
        before_board = [
            ["1536", "48", "12", "6"],
            ["3", "48", "48", "3"],
            [blue, red, "3", blue],
            ["3", ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, ws.TOKEN_EMPTY],
        ]
        observed_after_board = [
            ["1536", "48", "12", "6"],
            ["3", red, "48", "3"],
            [blue, ws.TOKEN_EMPTY, "3", blue],
            ["3", blue, ws.TOKEN_EMPTY, ws.TOKEN_EMPTY],
        ]
        gray_cells = self._gray_cells_from_board_fixture("move69_after_board.png")

        repair = sh.find_single_step_gray_repair(
            before_board,
            "blue",
            observed_after_board,
            gray_cells=gray_cells,
            max_gray_mismatches=1,
        )

        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertEqual(repair.step.direction, "up")
        self.assertEqual(repair.step.after_board[0][1], "96")
        self.assertEqual(repair.mismatch_positions, [(0, 1)])

    def test_unique_gray_repair_recovers_move47(self):
        blue = ws.SMALL_COLOR_MAP["blue"]
        red = ws.SMALL_COLOR_MAP["red"]
        before_board = [
            ["1536", "48", "24", "6"],
            [red, "48", "24", "3"],
            [ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, "6", blue],
            [ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, "3"],
        ]
        observed_after_board = [
            ["1536", "48", "48", "6"],
            [red, ws.TOKEN_EMPTY, "6", "3"],
            [ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, blue],
            [ws.TOKEN_EMPTY, ws.TOKEN_EMPTY, blue, "3"],
        ]
        gray_cells = self._gray_cells_from_board_fixture("move47_after_board.png")

        repair = sh.find_single_step_gray_repair(
            before_board,
            "blue",
            observed_after_board,
            gray_cells=gray_cells,
            max_gray_mismatches=2,
        )

        self.assertIsNotNone(repair)
        assert repair is not None
        self.assertEqual(repair.step.direction, "up")
        self.assertEqual(repair.step.after_board[0][1], "96")
        self.assertEqual(repair.step.inserted_pos, (3, 2))
        self.assertEqual(repair.mismatch_positions, [(0, 1)])


if __name__ == "__main__":
    unittest.main()

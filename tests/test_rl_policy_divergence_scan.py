import unittest

import numpy as np

from threes_rl.continue_from_replays import StartCase
from threes_rl.policy_divergence_scan import scan_cases, select_scan_cases, summarize_records
from threes_rl.sim import DOWN, LEFT, RIGHT, UP, SimState, preview_from_label


class FixedValuePolicy:
    def __init__(self, values):
        self.values = dict(values)

    def action_values(self, state, sim):
        legal = sim.legal_actions(state)
        return [(action, self.values[action]) for action in legal]


class SingleActionPolicy:
    def __init__(self, action: int) -> None:
        self.action = int(action)

    def __call__(self, state, sim, rng):
        return self.action


def scan_state(board=None) -> SimState:
    if board is None:
        board = [
            [1536, 3072, 0, 0],
            [768, 384, 192, 96],
            [48, 24, 12, 6],
            [3, 2, 1, 0],
        ]
    return SimState(
        board=np.asarray(board, dtype=np.int32),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=120,
        game_over=False,
    )


class PolicyDivergenceScanTests(unittest.TestCase):
    def _case(self, idx: int, phase: str, board=None) -> StartCase:
        return StartCase(
            id=f"case{idx}",
            state=scan_state(board),
            starter_tile=1536,
            source_replay="replay.json",
            source_seed=7,
            frame_index=idx,
            start_score=123,
            start_max_tile_excl_starter=3072,
            phase=phase,
        )

    def test_phase_balanced_selection_round_robins_phase_buckets(self):
        cases = [
            *(self._case(idx, "endgame_3072p") for idx in range(10)),
            self._case(20, "late_1536"),
            self._case(30, "mid_384_768"),
        ]

        selected = select_scan_cases(cases, max_states=6, seed=1, sample_mode="phase_balanced")

        phases = [case.phase for case in selected]
        self.assertEqual(phases.count("mid_384_768"), 1)
        self.assertEqual(phases.count("late_1536"), 1)
        self.assertEqual(phases.count("endgame_3072p"), 4)

    def test_stratum_balanced_selection_round_robins_phase_risk_buckets(self):
        low_risk_board = [
            [1536, 768, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        high_risk_board = [
            [1536, 3072, 1, 2],
            [768, 384, 192, 96],
            [48, 24, 12, 6],
            [3, 2, 1, 3],
        ]
        cases = [
            *(self._case(idx, "endgame_3072p", high_risk_board) for idx in range(8)),
            self._case(20, "mid_384_768", low_risk_board),
            self._case(30, "endgame_3072p", low_risk_board),
        ]

        selected = select_scan_cases(cases, max_states=5, seed=1, sample_mode="stratum_balanced")

        self.assertEqual(len(selected), 5)
        self.assertIn("mid_384_768", {case.phase for case in selected})
        self.assertGreaterEqual(
            sum(1 for case in selected if case.state.board[0, 1] == 768 and np.count_nonzero(case.state.board == 0) > 4),
            1,
        )

    def test_scan_cases_reports_changed_action_and_value_delta(self):
        case = StartCase(
            id="case",
            state=scan_state(),
            starter_tile=1536,
            source_replay="replay.json",
            source_seed=7,
            frame_index=12,
            start_score=123,
            start_max_tile_excl_starter=3072,
            phase="endgame_3072p",
        )
        base = FixedValuePolicy({UP: 10.0, DOWN: 5.0, LEFT: 4.0, RIGHT: 1.0})
        candidate = FixedValuePolicy({UP: 8.0, DOWN: 5.0, LEFT: 20.0, RIGHT: 1.0})

        records = scan_cases(base_policy=base, candidate_policy=candidate, cases=[case], include_values=True)
        summary = summarize_records(records)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].changed)
        self.assertEqual(records[0].base_action, "up")
        self.assertEqual(records[0].candidate_action, "left")
        self.assertEqual(records[0].corner_risk, "high_corner_risk")
        self.assertEqual(records[0].stratum, "endgame_3072p/high_corner_risk")
        self.assertEqual(records[0].base_top_two, ["up", "down"])
        self.assertEqual(records[0].candidate_top_two, ["left", "up"])
        self.assertTrue(records[0].top_two_changed)
        self.assertEqual(records[0].base_margin, 5.0)
        self.assertEqual(records[0].max_abs_value_delta, 16.0)
        self.assertEqual(summary["changed_actions"], 1)
        self.assertEqual(summary["changed_top_two"], 1)
        self.assertEqual(summary["by_phase"]["endgame_3072p"]["changed"], 1)
        self.assertEqual(summary["by_phase"]["endgame_3072p"]["top_two_changed"], 1)
        self.assertEqual(summary["by_stratum"]["endgame_3072p/high_corner_risk"]["changed"], 1)

    def test_scan_cases_reports_top_two_change_without_best_action_change(self):
        case = StartCase(
            id="case",
            state=scan_state(),
            starter_tile=1536,
            source_replay="replay.json",
            source_seed=7,
            frame_index=12,
            start_score=123,
            start_max_tile_excl_starter=3072,
            phase="endgame_3072p",
        )
        base = FixedValuePolicy({UP: 10.0, DOWN: 9.0, LEFT: 8.0, RIGHT: 1.0})
        candidate = FixedValuePolicy({UP: 10.0, DOWN: 8.0, LEFT: 9.0, RIGHT: 1.0})

        records = scan_cases(base_policy=base, candidate_policy=candidate, cases=[case])
        summary = summarize_records(records)

        self.assertFalse(records[0].changed)
        self.assertTrue(records[0].top_two_changed)
        self.assertEqual(records[0].base_top_two, ["up", "down"])
        self.assertEqual(records[0].candidate_top_two, ["up", "left"])
        self.assertEqual(summary["changed_actions"], 0)
        self.assertEqual(summary["changed_top_two"], 1)

    def test_scan_cases_does_not_report_top_two_change_for_single_action_wrapper(self):
        case = StartCase(
            id="case",
            state=scan_state(),
            starter_tile=1536,
            source_replay="replay.json",
            source_seed=7,
            frame_index=12,
            start_score=123,
            start_max_tile_excl_starter=3072,
            phase="endgame_3072p",
        )
        base = FixedValuePolicy({UP: 10.0, DOWN: 9.0, LEFT: 8.0, RIGHT: 1.0})
        candidate = SingleActionPolicy(UP)

        records = scan_cases(base_policy=base, candidate_policy=candidate, cases=[case])
        summary = summarize_records(records)

        self.assertFalse(records[0].changed)
        self.assertFalse(records[0].top_two_changed)
        self.assertEqual(records[0].candidate_top_two, ["up"])
        self.assertEqual(summary["changed_top_two"], 0)


if __name__ == "__main__":
    unittest.main()

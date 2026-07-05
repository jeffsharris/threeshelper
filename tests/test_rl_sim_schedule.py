import collections
import unittest

import numpy as np

import state_hunt as sh
import window_stream as ws
from threes_rl.sim import ThreesSim, board_to_tokens, label_for_insert_value


class SimScheduleTests(unittest.TestCase):
    def test_lockstep_tile_cycle_matches_tracker_updates(self):
        rng = np.random.default_rng(900)
        checked = 0
        for episode in range(100):
            sim = ThreesSim(np.random.default_rng(int(rng.integers(1_000_000))), starter_tile=1536)
            state = sim.reset()
            while not state.game_over and state.move_count < 200:
                before_snapshot = sim.tile_cycle_snapshot(state)
                actions = sim.legal_actions(state)
                action = int(actions[int(rng.integers(len(actions)))])
                state, info = sim.step(state, action)
                if info.terminal_merge:
                    break
                check = sh.preview_check_from_snapshot(
                    before_snapshot,
                    board_to_tokens(state.board),
                    state.preview.label,
                    inserted_values=[info.inserted_value],
                )
                self.assertTrue(check["valid"], check.get("reason"))
                self.assertEqual(check["next_snapshot"], sim.tile_cycle_snapshot(state))
                checked += 1
        self.assertGreaterEqual(checked, 1000)

    def test_small_bags_have_exact_4_4_4_counts(self):
        sim = ThreesSim(np.random.default_rng(123))
        counts = {"red": 4, "blue": 4, "gray": 4}
        small_pos = 0
        small_seen_total = 0
        span_small_pos = 0
        large_pending = False
        max_tile = 0
        bag_counter = collections.Counter()
        complete_bags = 0

        while complete_bags < 200:
            preview = sim._sample_preview(counts, small_pos, small_seen_total, span_small_pos, large_pending, max_tile)
            self.assertNotEqual(preview.label, "large_candidates")
            bag_counter[preview.label] += 1
            counts, small_pos, small_seen_total, span_small_pos, large_pending = sim._consume_preview(
                counts, small_pos, small_seen_total, span_small_pos, large_pending, preview.label
            )
            if small_pos == 0:
                self.assertEqual(bag_counter, collections.Counter({"red": 4, "blue": 4, "gray": 4}))
                bag_counter.clear()
                complete_bags += 1

    def test_large_probability_matches_tracker_formula(self):
        sim = ThreesSim(np.random.default_rng(1))
        counts = {"red": 4, "blue": 4, "gray": 4}
        for span_pos in range(0, 21):
            options = sim.preview_options(
                counts,
                small_pos=0,
                small_seen_total=21 + span_pos,
                span_small_pos=span_pos,
                large_pending=True,
                max_tile=1536,
            )
            large_prob = sum(option.probability for option in options if option.preview.label == "large_candidates")
            self.assertAlmostEqual(large_prob, 1.0 / (21 - span_pos))

    def test_preview_distribution_agrees_with_tile_cycle_for_labels(self):
        sim = ThreesSim(np.random.default_rng(1))
        cycle = ws.TileCycle()
        cycle.small_counts = {"red": 2, "blue": 3, "gray": 1}
        cycle.small_pos = 6
        cycle.small_seen_total = 30
        cycle.span_small_pos = 5
        cycle.large_pending = True
        cycle.max_tile = 1536
        options = sim.preview_options(
            cycle.small_counts,
            cycle.small_pos,
            cycle.small_seen_total,
            cycle.span_small_pos,
            cycle.large_pending,
            cycle.max_tile,
        )
        label_probs = collections.defaultdict(float)
        for option in options:
            label_probs[option.preview.label] += option.probability
        self.assertEqual(set(label_probs), {"red", "blue", "gray", "large_candidates"})
        for label, prob in cycle.probabilities().items():
            self.assertAlmostEqual(label_probs[label], prob)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from threes_rl.record_replay import record_replay_for_policy
from threes_rl.replay_provenance import ORIGIN_FRESH, ORIGIN_REPLAY_START, replay_provenance


def first_legal_policy(state, sim, _rng):
    return sim.legal_actions(state)[0]


class ReplayProvenanceTests(unittest.TestCase):
    def test_recorded_policy_replay_has_fresh_root_provenance(self):
        replay = record_replay_for_policy(first_legal_policy, "fixture_policy", 123, 1536, 2)

        provenance = replay_provenance(replay, "fixture/replay.json")

        self.assertEqual(replay["replay_origin"], ORIGIN_FRESH)
        self.assertEqual(provenance["replay_origin"], ORIGIN_FRESH)
        self.assertEqual(provenance["root_origin"], ORIGIN_FRESH)
        self.assertTrue(provenance["replay_reset_invariant"])
        first_board = np.asarray(replay["frames"][0]["state"]["board"])
        self.assertEqual(int(first_board[0, 0]), 1536)

    def test_old_replay_start_is_not_mistaken_for_fresh_root(self):
        replay = {
            "policy": "train_td:mixed",
            "seed": 7,
            "starter_tile": 1536,
            "training_config": {"start_state_prob": 0.4},
            "frames": [
                {
                    "index": 0,
                    "state": {
                        "move_count": 408,
                        "board": [[1536, 3072, 0, 0], [768, 384, 192, 96], [48, 24, 12, 6], [3, 2, 1, 0]],
                        "score": 238662,
                        "max_tile": 3072,
                        "game_over": False,
                        "preview": {"kind": "gray", "label": "gray", "value": 3, "candidates": []},
                        "tile_cycle": {
                            "small_counts": {"red": 4, "blue": 4, "gray": 4},
                            "small_pos": 8,
                            "small_seen_total": 0,
                            "span_small_pos": 0,
                            "large_pending": False,
                            "max_tile": 3072,
                        },
                    },
                    "move": None,
                }
            ],
        }

        provenance = replay_provenance(replay, "mixed/replay.json")

        self.assertEqual(provenance["replay_origin"], ORIGIN_REPLAY_START)
        self.assertEqual(provenance["root_origin"], "unknown")
        self.assertFalse(provenance["root_is_genuine"])
        self.assertFalse(provenance["replay_reset_invariant"])


if __name__ == "__main__":
    unittest.main()

import unittest

from threes_rl.compare_replays import first_divergence, summarize_replay


def frame(index: int, action: str | None = None, board_value: int = 0) -> dict:
    payload = {
        "index": index,
        "state": {
            "move_count": index,
            "board": [[board_value, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "score": board_value,
            "preview": {"kind": "red", "label": "red", "value": 2, "candidates": []},
            "legal_actions": ["left", "right"],
        },
    }
    if action is not None:
        payload["move"] = {"action": action, "score_after": board_value}
    else:
        payload["move"] = None
    return payload


class CompareReplaysTests(unittest.TestCase):
    def test_first_divergence_reports_action_choice(self):
        left = {"frames": [frame(0), frame(1, "left", 1), frame(2, "up", 2)]}
        right = {"frames": [frame(0), frame(1, "right", 1), frame(2, "up", 2)]}

        divergence = first_divergence(left, right)

        self.assertEqual(divergence["kind"], "action")
        self.assertEqual(divergence["move_number"], 1)
        self.assertEqual(divergence["left_action"], "left")
        self.assertEqual(divergence["right_action"], "right")

    def test_first_divergence_reports_state_mismatch(self):
        left = {"frames": [frame(0), frame(1, "left", 1)]}
        right = {"frames": [frame(0, board_value=3), frame(1, "left", 1)]}

        divergence = first_divergence(left, right)

        self.assertEqual(divergence["kind"], "state_mismatch")
        self.assertEqual(divergence["move_number"], 1)

    def test_first_divergence_reports_no_difference(self):
        left = {"frames": [frame(0), frame(1, "left", 1)]}
        right = {"frames": [frame(0), frame(1, "left", 1)]}

        divergence = first_divergence(left, right)

        self.assertEqual(divergence["kind"], "none")

    def test_summarize_replay_uses_final_frame(self):
        replay = {
            "policy": "policy-a",
            "seed": 1000,
            "starter_tile": 1536,
            "frames": [frame(0), frame(1, "left", 12)],
        }

        summary = summarize_replay(replay)

        self.assertEqual(summary["policy"], "policy-a")
        self.assertEqual(summary["final_score"], 12)
        self.assertEqual(summary["final_board"][0][0], 12)


if __name__ == "__main__":
    unittest.main()

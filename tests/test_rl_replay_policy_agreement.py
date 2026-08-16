import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.replay_policy_agreement import scan_replays, scan_state_records, summarize_records
from threes_rl.sim import DOWN, LEFT, RIGHT, UP, SimState, ThreesSim, preview_from_label


class FixedValuePolicy:
    def __init__(self, values):
        self.values = dict(values)

    def action_values(self, state, sim):
        return [(action, self.values[action]) for action in sim.legal_actions(state)]


def agreement_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 3072, 0, 0],
                [768, 384, 192, 96],
                [48, 24, 12, 6],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("blue"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=120,
        game_over=False,
    )


def write_replay(path: Path, recorded_action: str = "left") -> None:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = agreement_state()
    replay = {
        "policy": "human_observed",
        "seed": 7,
        "starter_tile": 1536,
        "frames": [
            {"index": 0, "state": state_payload(state, sim), "move": None},
            {"index": 1, "state": state_payload(state, sim), "move": {"action": recorded_action}},
        ],
    }
    path.write_text(json.dumps(replay))


class ReplayPolicyAgreementTests(unittest.TestCase):
    def test_scan_reports_recorded_action_rank_and_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            write_replay(replay_path, recorded_action="left")
            policy = FixedValuePolicy({UP: 10.0, DOWN: 7.0, LEFT: 4.0, RIGHT: 1.0})

            payload = scan_replays(
                policy=policy,
                policy_spec="fixed",
                replay_paths=[replay_path],
                min_tile=3072,
                high_confidence_margin=0.10,
            )

        self.assertEqual(payload["summary"]["records"], 1)
        self.assertEqual(payload["summary"]["action_matches"], 0)
        self.assertEqual(payload["summary"]["high_confidence_misses"], 1)
        record = payload["records"][0]
        self.assertEqual(record["recorded_action"], "left")
        self.assertEqual(record["policy_action"], "up")
        self.assertEqual(record["recorded_rank"], 3)
        self.assertFalse(record["recorded_in_top_two"])
        self.assertEqual(record["value_gap_to_recorded"], 6.0)
        self.assertAlmostEqual(record["normalized_value_gap_to_recorded"], 0.6)

    def test_summarize_counts_top_two_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            write_replay(replay_path, recorded_action="left")
            policy = FixedValuePolicy({LEFT: 10.0, UP: 7.0, DOWN: 4.0, RIGHT: 1.0})

            payload = scan_replays(policy=policy, policy_spec="fixed", replay_paths=[replay_path], min_tile=3072)

        self.assertEqual(payload["summary"]["action_matches"], 1)
        self.assertEqual(payload["summary"]["recorded_in_top_two"], 1)
        self.assertEqual(payload["summary"]["recorded_rank_counts"], {"1": 1})
        self.assertEqual(payload["summary"]["by_phase"]["endgame_3072p"]["action_matches"], 1)

    def test_scan_state_records_uses_source_next_action(self):
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        state = agreement_state()
        record = {
            "id": "transition-record",
            "source_replay": "human.json",
            "source_seed": 99,
            "source_frame_index": 12,
            "starter_tile": 1536,
            "source_next_action": "left",
            "state": state_payload(state, sim),
        }
        policy = FixedValuePolicy({UP: 10.0, DOWN: 7.0, LEFT: 4.0, RIGHT: 1.0})

        payload = scan_state_records(
            policy=policy,
            policy_spec="fixed",
            state_records=[record],
            min_tile=3072,
            high_confidence_margin=0.10,
        )

        self.assertEqual(payload["summary"]["records"], 1)
        self.assertEqual(payload["summary"]["scanned_state_records"], 1)
        self.assertEqual(payload["records"][0]["id"], "transition-record")
        self.assertEqual(payload["records"][0]["recorded_action"], "left")
        self.assertEqual(payload["records"][0]["recorded_rank"], 3)


if __name__ == "__main__":
    unittest.main()

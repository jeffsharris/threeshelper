import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.pre_nearfail_support_action_labels import run_labeling
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def label_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 768, 384, 0],
                [192, 0, 0, 0],
                [0, 0, 0, 0],
                [3, 6, 12, 24],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=40,
        span_small_pos=0,
        large_pending=False,
        max_tile=1536,
        move_count=80,
        game_over=False,
    )


class PreNearfailSupportActionLabelsTests(unittest.TestCase):
    def test_run_labeling_emits_support_and_raw2_endpoints(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
            state = label_state()
            records_path = tmp_path / "records.json"
            records_path.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "id": "case-pre5",
                                "kind": "pre_nearfail_support_window_state",
                                "window_offset": 5,
                                "source_next_action": "left",
                                "source_replay": "fixture/replay.json",
                                "source_seed": 7,
                                "source_frame_index": 80,
                                "root_origin": "fresh",
                                "root_seed": 7,
                                "root_replay": "fixture/root.json",
                                "starter_tile": 1536,
                                "support_bucket": "one_384",
                                "state": state_payload(state, sim),
                            }
                        ]
                    }
                )
            )

            payload = run_labeling(
                records_json=[records_path],
                policy_name="greedy",
                support_horizons=(1,),
                raw2_horizons=(1,),
                repeats_per_block=1,
                seed_blocks=(101, 202),
                max_starts=0,
                seed=303,
                offsets={5},
                root_origins={"fresh"},
                default_starter_tile=1536,
                first_action_mode="all",
                base_action_mode="recorded",
                out_dir=tmp_path / "out",
                progress_every=0,
            )

            summary = payload["summary"]
            self.assertEqual(summary["cases_selected"], 1)
            self.assertEqual(summary["unique_roots"], 1)
            self.assertEqual(summary["seed_blocks"], [0, 1])
            self.assertIn("support_h1", summary["winner_stability"])
            self.assertIn("raw2_h1", summary["winner_stability"])
            self.assertGreater(len(payload["action_summary"]), 0)
            self.assertTrue((tmp_path / "out" / "pre_nearfail_support_action_labels.json").exists())
            self.assertTrue((tmp_path / "out" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()

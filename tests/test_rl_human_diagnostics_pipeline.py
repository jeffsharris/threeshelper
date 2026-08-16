import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.human_diagnostics_pipeline import run_pipeline
from threes_rl.sim import board_to_tokens, simulate_base_move


def make_events() -> list[dict[str, object]]:
    before = np.asarray(
        [
            [1536, 0, 1, 0],
            [2, 0, 3, 0],
            [1, 0, 2, 0],
            [3, 1, 2, 0],
        ],
        dtype=np.int32,
    )
    shifted, eligible = simulate_base_move(before, "left")
    after = shifted.copy()
    after[0, 3] = 1
    return [
        {
            "type": "game_start",
            "game_index": 1,
            "board": board_to_tokens(before),
            "preview_label": "blue",
        },
        {
            "type": "observed_move",
            "game_index": 1,
            "move_index_start": 1,
            "move_index": 1,
            "before_board": board_to_tokens(before),
            "before_preview_label": "blue",
            "after_board": board_to_tokens(after),
            "after_preview_label": "red",
            "unknown_board": False,
            "unknown_preview": False,
            "preview_check": {"valid": True},
            "transition_check": {
                "valid": True,
                "inserted_value": 1,
                "inserted_pos": [0, 3],
                "eligible_positions": [list(pos) for pos in eligible],
                "expected_values": [1],
            },
            "direction": "left",
            "direction_sequence": ["left"],
            "step_count": 1,
            "transition_path": [
                {
                    "direction": "left",
                    "preview_label": "blue",
                    "inserted_value": 1,
                    "inserted_pos": [0, 3],
                    "eligible_positions": [list(pos) for pos in eligible],
                    "expected_values": [1],
                    "after_board": board_to_tokens(after),
                }
            ],
        },
    ]


def args_for(events_path: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        events_jsonl=[[events_path]],
        out_dir=out_dir,
        starter="1536",
        min_valid_moves=1,
        no_replay_html=False,
        policy=None,
        policy_file=None,
        min_tile=3,
        phase_filter=["early"],
        corner_risk_filter=None,
        reservoir_first_per="none",
        reservoir_max_records=10,
        reservoir_max_per_stratum=0,
        reservoir_sort_by="source",
        no_transition_windows=False,
        transition_targets="3,1536",
        transition_window_size=40,
        transition_max_records=0,
        transition_no_failures=False,
        no_support_ladder=False,
        support_ladder_targets="raw_duplicate_768,raw_one_1536",
        support_ladder_window_size=40,
        support_ladder_max_records=0,
        support_ladder_no_failures=False,
        sample_mode="top-two",
        margin_threshold=0.002,
        min_top_value=0.0,
        scan_max_samples=24,
        scan_max_per_stratum=4,
        scan_first_per="replay-stratum",
        max_state_records=0,
        scan_phase_filter=["early"],
        scan_corner_risk_filter=None,
        anchor_min_tile=1536,
        geometry_min_tile=1536,
        geometry_min_delta=1.0,
        action_value_cache=None,
        agreement_min_tile=None,
        agreement_phase_filter=None,
        agreement_corner_risk_filter=None,
        agreement_max_records=0,
        agreement_high_confidence_margin=0.01,
        no_agreement_report=False,
        progress_every_state_records=0,
        label_seed=20260706,
        stability_threshold=0.70,
        label_workers=1,
    )


class HumanDiagnosticsPipelineTests(unittest.TestCase):
    def test_pipeline_imports_replay_and_builds_reservoir(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            out_dir = tmp_path / "diagnostics"
            events_path.write_text("\n".join(json.dumps(event) for event in make_events()))

            payload = run_pipeline(args_for(events_path, out_dir))

            self.assertEqual(payload["games_imported"], 1)
            self.assertEqual(payload["games_skipped"], 0)
            self.assertEqual(len(payload["replay_json"]), 1)
            self.assertGreaterEqual(payload["reservoir"]["summary"]["records"], 1)
            self.assertTrue((out_dir / "human_diagnostics_manifest.json").exists())
            self.assertTrue((out_dir / "human_diagnostics.html").exists())
            self.assertTrue((out_dir / "reservoir" / "records.json").exists())
            self.assertTrue((out_dir / "transition_windows" / "records.json").exists())
            self.assertTrue((out_dir / "support_ladder_windows" / "records.json").exists())
            self.assertIn("records", payload["transition_windows"]["summary"])
            self.assertEqual(payload["support_ladder"]["summary"]["root_origin_filter"], ["human"])
            self.assertEqual(payload["next_steps"], [])
            self.assertEqual(payload["agreement"], {})

    def test_pipeline_can_run_scan_only_diagnostic(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            out_dir = tmp_path / "diagnostics_scan"
            events_path.write_text("\n".join(json.dumps(event) for event in make_events()))
            args = args_for(events_path, out_dir)
            args.policy = "corner2"
            args.margin_threshold = 1.0
            args.scan_first_per = "none"

            payload = run_pipeline(args)

            self.assertEqual(payload["games_imported"], 1)
            self.assertEqual(payload["agreement"]["summary"]["records"], 1)
            self.assertIn("scan_stats", payload["scan"]["summary"])
            self.assertEqual(payload["scan"]["summary"]["scan_stats"]["accepted_samples"], 2)
            self.assertTrue((out_dir / "top_two_scan" / "swing_labels.json").exists())
            self.assertTrue((out_dir / "policy_agreement" / "agreement.json").exists())
            self.assertEqual(len(payload["next_steps"]), 1)
            self.assertEqual(payload["next_steps"][0]["name"], "cheap_label_pilot")

    def test_pipeline_accepts_policy_file_for_scan(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            events_path = tmp_path / "events.jsonl"
            policy_path = tmp_path / "policy.txt"
            out_dir = tmp_path / "diagnostics_policy_file"
            events_path.write_text("\n".join(json.dumps(event) for event in make_events()))
            policy_path.write_text("# current actor\ncorner2\n")
            args = args_for(events_path, out_dir)
            args.policy_file = policy_path
            args.margin_threshold = 1.0
            args.scan_first_per = "none"

            payload = run_pipeline(args)

            self.assertEqual(payload["policy"], "corner2")
            self.assertEqual(payload["agreement"]["summary"]["records"], 1)
            self.assertEqual(payload["scan"]["summary"]["scan_stats"]["accepted_samples"], 2)
            self.assertEqual(len(payload["next_steps"]), 1)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.mcts_rollout_gate import mcts_action, run_gate
from threes_rl.rare_event_frontier import case_from_record
from threes_rl.record_replay import state_payload
from threes_rl.sim import SimState, ThreesSim, preview_from_label


def pre_1536_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 768, 768, 0],
                [384, 192, 96, 48],
                [24, 12, 6, 3],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=8,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=1536,
        move_count=80,
        game_over=False,
    )


def record_payload() -> dict:
    sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
    state = pre_1536_state()
    return {
        "id": "mcts-gate-case",
        "starter_tile": 1536,
        "source_replay": "fresh/source.json",
        "source_seed": 7,
        "source_frame_index": 80,
        "source_policy": "fixture_policy",
        "source_next_action": "left",
        "root_origin": "fresh",
        "root_replay": "fresh/root.json",
        "root_seed": 7,
        "root_frame_index": 0,
        "root_policy": "fixture_policy",
        "state": state_payload(state, sim),
    }


class MctsRolloutGateTests(unittest.TestCase):
    def test_mcts_action_returns_legal_action_and_summary(self):
        record = record_payload()
        case = case_from_record(record, default_starter_tile=1536)
        assert case is not None
        policy = lambda state, sim, rng: sim.legal_actions(state)[0]

        action, info = mcts_action(
            policy=policy,
            case=case,
            simulations=4,
            depth=2,
            seed=123,
            exploration=10.0,
        )

        self.assertIn(action, ThreesSim(np.random.default_rng(1), starter_tile=1536).legal_actions(case.state))
        self.assertEqual(info["simulations"], 4)
        self.assertGreater(info["root_visits"], 0)

    def test_mcts_action_supports_target_reward_mode(self):
        record = record_payload()
        case = case_from_record(record, default_starter_tile=1536)
        assert case is not None
        policy = lambda state, sim, rng: sim.legal_actions(state)[0]

        action, info = mcts_action(
            policy=policy,
            case=case,
            simulations=4,
            depth=2,
            seed=124,
            exploration=10.0,
            target="reached_1536",
            reward_mode="target",
            leaf_weight=0.0,
        )

        self.assertIn(action, ThreesSim(np.random.default_rng(1), starter_tile=1536).legal_actions(case.state))
        self.assertEqual(info["reward_mode"], "target")
        self.assertEqual(info["leaf_weight"], 0.0)

    def test_run_gate_writes_target_score_and_survival_lifts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            records_path.write_text(json.dumps({"records": [record_payload()]}))

            payload = run_gate(
                records_json=[records_path],
                policy_name="greedy",
                target="reached_1536",
                horizon=1,
                eval_repeats=1,
                eval_blocks=2,
                max_starts=0,
                seed=456,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                simulations=4,
                depth=2,
                exploration=10.0,
                mcts_select_by="visits",
                out_dir=out_dir,
                progress_every=0,
                min_promotion_roots=20,
            )

            self.assertTrue((out_dir / "mcts_rollout_gate.json").exists())
            self.assertTrue((out_dir / "report.md").exists())

        summary = payload["summary"]
        self.assertEqual(summary["target"], "reached_1536")
        self.assertEqual(summary["cases_selected"], 1)
        self.assertEqual(summary["unique_roots"], 1)
        self.assertIn("mean_eval_score_lift_vs_base", summary)
        self.assertIn("mean_eval_survival_lift_vs_base", summary)
        self.assertEqual(summary["mcts_reward_mode"], "value")
        self.assertFalse(summary["promotion_screen"]["enough_roots"])

    def test_run_gate_resumes_checkpointed_eval_rollouts(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            records_path = tmp_path / "records.json"
            out_dir = tmp_path / "out"
            records_path.write_text(json.dumps({"records": [record_payload()]}))

            payload1 = run_gate(
                records_json=[records_path],
                policy_name="greedy",
                target="reached_1536",
                horizon=1,
                eval_repeats=1,
                eval_blocks=2,
                max_starts=0,
                seed=789,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                simulations=4,
                depth=2,
                exploration=10.0,
                mcts_select_by="visits",
                out_dir=out_dir,
                progress_every=0,
                min_promotion_roots=20,
                checkpoint_rollouts=True,
            )
            payload2 = run_gate(
                records_json=[records_path],
                policy_name="greedy",
                target="reached_1536",
                horizon=1,
                eval_repeats=1,
                eval_blocks=2,
                max_starts=0,
                seed=789,
                root_origins={"fresh"},
                case_ids=None,
                default_starter_tile=1536,
                simulations=4,
                depth=2,
                exploration=10.0,
                mcts_select_by="visits",
                out_dir=out_dir,
                progress_every=0,
                min_promotion_roots=20,
                checkpoint_rollouts=True,
            )
            progress_exists = (out_dir / "mcts_gate_progress.json").exists()

        self.assertTrue(progress_exists)
        self.assertGreater(payload1["summary"]["rollouts_ran"], 0)
        self.assertEqual(payload1["summary"]["rollouts_resumed"], 0)
        self.assertEqual(payload2["summary"]["rollouts_ran"], 0)
        self.assertEqual(payload2["summary"]["rollouts_resumed"], payload1["summary"]["rollouts_planned"])


if __name__ == "__main__":
    unittest.main()

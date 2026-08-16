import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.human_play_server import HumanGameSession, load_restart_catalog
from threes_rl.record_replay import state_payload
from threes_rl.sim import ThreesSim, preview_from_label
from threes_rl.train_td import state_from_replay_payload


class HumanPlayServerTests(unittest.TestCase):
    class StubAdvisor:
        policy_file_sha256 = "advisor-hash"

        def metadata(self):
            return {
                "mode": "incumbent_recommendation_visible",
                "policy_file_sha256": self.policy_file_sha256,
                "status": "ready",
            }

        def recommend(self, state, sim, *, decision_seed):
            action = int(sim.legal_actions(state)[0])
            return {
                "status": "ok",
                "action": ["up", "down", "left", "right"][action],
                "action_index": action,
                "action_values": [],
                "policy_file_sha256": self.policy_file_sha256,
                "decision_seed": decision_seed,
            }

    def test_session_records_exact_replay_after_each_move(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            session = HumanGameSession(
                Path(tmp),
                logical_seed=123,
                deck_stream_id=456,
                slot_stream_id=789,
                player_id="test_player",
            )
            action = session.sim.legal_actions(session.state)[0]
            result = session.move(action, decision_time_ms=321.5)

            self.assertTrue(result["moved"])
            self.assertEqual(result["recorded_frames"], 2)
            replay = json.loads(session.replay_path.read_text())
            self.assertEqual(replay["replay_origin"], "human")
            self.assertEqual(replay["root_origin"], "human")
            self.assertEqual(replay["policy"], "human_web")
            self.assertEqual(replay["player_id"], "test_player")
            self.assertEqual(replay["stream_metadata"]["deck_stream_id"], 456)
            self.assertEqual(replay["stream_metadata"]["slot_stream_id"], 789)
            self.assertEqual(len(replay["frames"]), 2)
            self.assertEqual(replay["frames"][1]["move"]["decision_time_ms"], 321.5)

            restored = state_from_replay_payload(replay["frames"][1]["state"])
            self.assertEqual(restored.move_count, session.state.move_count)
            self.assertEqual(restored.preview, session.state.preview)
            self.assertEqual(restored.small_counts, session.state.small_counts)

    def test_illegal_move_does_not_create_frame(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            session = HumanGameSession(
                Path(tmp),
                logical_seed=1,
                deck_stream_id=2,
                slot_stream_id=3,
            )
            session.state = session.sim.state_from_snapshot(
                np.asarray(
                    [
                        [1536, 0, 0, 0],
                        [3, 0, 0, 0],
                        [6, 0, 0, 0],
                        [12, 0, 0, 0],
                    ],
                    dtype=np.int32,
                ),
                preview_from_label("blue"),
                ({"red": 2, "blue": 2, "gray": 2}, 6, 0, 0, False, 1536),
            )

            result = session.move("left")

            self.assertFalse(result["moved"])
            self.assertEqual(result["recorded_frames"], 1)

    def test_visible_recommendation_is_returned_and_recorded_with_agreement(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            session = HumanGameSession(
                Path(tmp),
                logical_seed=4,
                deck_stream_id=5,
                slot_stream_id=6,
                advisor=self.StubAdvisor(),
            )
            recommendation = session.public_payload()["recommendation"]

            result = session.move(recommendation["action"])
            replay = json.loads(session.replay_path.read_text())
            move = replay["frames"][1]["move"]

            self.assertTrue(result["moved"])
            self.assertEqual(move["recommended_action"], recommendation["action"])
            self.assertTrue(move["human_model_agreement"])
            self.assertEqual(move["model_recommendation"], recommendation)
            self.assertTrue(replay["human_input"]["model_assistance_visible"])
            self.assertEqual(replay["model_assistance"]["policy_file_sha256"], "advisor-hash")

    def test_finish_writes_browser_replay(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            session = HumanGameSession(
                Path(tmp),
                logical_seed=11,
                deck_stream_id=12,
                slot_stream_id=13,
            )

            result = session.finish()

            self.assertEqual(result["status"], "ended_by_player")
            self.assertTrue(session.html_path.exists())
            self.assertIn("Threes Replay", session.html_path.read_text())

    def test_quality_annotation_updates_metadata_without_changing_frames(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            session = HumanGameSession(
                Path(tmp), logical_seed=14, deck_stream_id=15, slot_stream_id=16
            )
            before = json.loads(session.replay_path.read_text())

            result = session.annotate_quality("mistakes")
            after = json.loads(session.replay_path.read_text())

            self.assertEqual(result["quality_annotation"], "mistakes")
            self.assertEqual(after["quality_annotation"], "mistakes")
            self.assertEqual(after["frames"], before["frames"])
            with self.assertRaisesRegex(ValueError, "Unsupported quality"):
                session.annotate_quality("excellent")

    def test_restart_catalog_and_session_preserve_exact_state_and_provenance(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            source = HumanGameSession(
                root / "source", logical_seed=41, deck_stream_id=42, slot_stream_id=43
            )
            action = source.sim.legal_actions(source.state)[0]
            source.move(action)
            source_payload = json.loads(source.replay_path.read_text())
            source_state = source_payload["frames"][1]["state"]
            source_hash = hashlib.sha256(source.replay_path.read_bytes()).hexdigest()
            manifest_path = root / "restart_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "roots": [
                            {
                                "root_id": "restart-case",
                                "ancestry_cluster": source.session_id,
                                "role": "development",
                                "starter_tile": 1536,
                                "source_replay": str(source.replay_path),
                                "source_replay_sha256": source_hash,
                                "source_frame_index": 1,
                                "source_move_count": source_state["move_count"],
                                "state": source_state,
                            }
                        ]
                    }
                )
            )

            catalog = load_restart_catalog(manifest_path)
            restart = HumanGameSession(
                root / "continuation",
                logical_seed=51,
                deck_stream_id=52,
                slot_stream_id=53,
                restart_record=catalog["restart-case"],
            )
            replay = json.loads(restart.replay_path.read_text())

            self.assertEqual(replay["frames"][0]["state"], source_state)
            self.assertEqual(replay["replay_origin"], "continuation")
            self.assertEqual(replay["root_origin"], "human")
            self.assertEqual(replay["source_frame_index"], 1)
            self.assertFalse(replay["dashboard_eligible"])
            self.assertFalse(replay["dashboard_record_eligible"])
            self.assertEqual(
                replay["restart_metadata"]["source_stream_metadata"]["deck_stream_id"], 42
            )
            self.assertEqual(
                replay["restart_metadata"]["continuation_stream_metadata"]["deck_stream_id"], 52
            )
            self.assertEqual(restart.public_payload()["restart_root_id"], "restart-case")

    def test_play_page_supports_arrow_keys_and_bonus_preview(self):
        html = Path("threes_rl/human_play.html").read_text()

        self.assertIn("ArrowUp", html)
        self.assertIn('preview.kind === "bonus"', html)
        self.assertIn("preview.candidates.map", html)
        self.assertIn("preview-candidate", html)
        self.assertIn("Bonus candidates", html)
        self.assertIn("threesHumanPlaySession", html)
        self.assertIn("Small-tile bag", html)
        self.assertIn("Plus-block window", html)
        self.assertIn("cycleAfterVisible", html)
        self.assertIn("plusProbability", html)
        self.assertIn('data-quality="good"', html)
        self.assertIn("restart_root_id", html)
        self.assertIn("move-button.recommended", html)
        self.assertIn("incumbent recommendation", html)

    def test_bonus_preview_and_cycle_round_trip_without_loss(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            session = HumanGameSession(
                Path(tmp),
                logical_seed=21,
                deck_stream_id=22,
                slot_stream_id=23,
            )
            session.state = session.sim.state_from_snapshot(
                session.state.board,
                preview_from_label("large_candidates", (96, 192, 384)),
                ({"red": 2, "blue": 1, "gray": 3}, 6, 21, 7, True, 1536),
            )
            session.frames[0]["state"] = state_payload(session.state, session.sim)
            session._persist()

            replay = json.loads(session.replay_path.read_text())
            saved = replay["frames"][0]["state"]
            restored = state_from_replay_payload(saved)

            self.assertEqual(saved["preview"]["candidates"], [96, 192, 384])
            self.assertEqual(restored.preview.candidates, (96, 192, 384))
            self.assertEqual(restored.small_counts, {"red": 2, "blue": 1, "gray": 3})
            self.assertEqual(restored.small_pos, 6)
            self.assertEqual(restored.small_seen_total, 21)
            self.assertEqual(restored.span_small_pos, 7)
            self.assertTrue(restored.large_pending)

    def test_plus_probability_tracks_cycle_position(self):
        sim = ThreesSim.from_stream_ids(deck_stream_id=31, slot_stream_id=32, starter_tile=1536)
        state = sim.reset()

        early = sim._large_probability(20, 0, True, state.max_tile)
        first_pending = sim._large_probability(21, 0, True, state.max_tile)
        final_pending = sim._large_probability(21, 20, True, state.max_tile)
        not_pending = sim._large_probability(21, 20, False, state.max_tile)

        self.assertEqual(early, 0.0)
        self.assertAlmostEqual(first_pending, 1.0 / 21.0)
        self.assertEqual(final_pending, 1.0)
        self.assertEqual(not_pending, 0.0)


if __name__ == "__main__":
    unittest.main()

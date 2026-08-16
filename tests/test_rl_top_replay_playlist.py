import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from threes_rl.dashboard import collect_global_top_replays
from threes_rl.top_replay_playlist import build_top_replay_playlist, playlist_is_current, sync_top_replay_playlist


def _replay_payload(seed: int, score: int) -> dict[str, object]:
    return {
        "policy": "test_policy",
        "seed": seed,
        "starter_tile": 1536,
        "max_moves": 1,
        "final_score": score,
        "final_moves": 1,
        "final_max_tile": 1536,
        "game_over": False,
        "frames": [
            {
                "index": 0,
                "state": {
                    "move_count": 0,
                    "board": [[1536, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    "score": 59049,
                    "max_tile": 1536,
                    "game_over": False,
                    "preview": {"kind": "small", "label": "blue", "value": 1, "candidates": [1]},
                    "legal_actions": ["left"],
                    "legal_mask": [True, False, False, False],
                    "tile_cycle": {
                        "small_counts": {"1": 0, "2": 0, "3": 0},
                        "small_pos": 0,
                        "small_seen_total": 0,
                        "span_small_pos": 0,
                        "large_pending": False,
                        "max_tile": 1536,
                    },
                },
                "move": None,
            }
        ],
    }


def _write_top_game(root: Path, run: str, *, score: int, seed: int) -> None:
    replay_dir = root / "eval_artifacts" / run / "top_games" / f"rank_01_score_{score}_seed_{seed}_starter_1536"
    replay_dir.mkdir(parents=True)
    (replay_dir / "replay.json").write_text(json.dumps(_replay_payload(seed, score)))
    (replay_dir / "replay.html").write_text("<html></html>")
    summary = root / "eval_artifacts" / run / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "games": 1,
                "top_games": [
                    {
                        "seed": seed,
                        "starter_tile": 1536,
                        "score": score,
                        "score_minus_starter": score - 59049,
                        "moves": seed,
                        "max_tile": 1536,
                        "max_tile_excl_starter": 768,
                        "json": f"threes_rl/runs/eval_artifacts/{run}/top_games/{replay_dir.name}/replay.json",
                        "html": f"threes_rl/runs/eval_artifacts/{run}/top_games/{replay_dir.name}/replay.html",
                    }
                ],
            }
        )
    )


class TopReplayPlaylistTests(unittest.TestCase):
    def test_build_playlist_copies_global_top_three_to_stable_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_top_game(root, "low", score=100, seed=1)
            _write_top_game(root, "mid", score=200, seed=2)
            _write_top_game(root, "high", score=300, seed=3)
            _write_top_game(root, "record", score=400, seed=4)

            payload = build_top_replay_playlist(root=root, out_dir=root / "replays" / "top3", limit=3)

            self.assertEqual([item["score"] for item in payload["replays"]], [400, 300, 200])
            self.assertEqual(payload["copied_count"], 3)
            self.assertTrue((root / "replays" / "top3" / "manifest.json").exists())
            self.assertTrue((root / "replays" / "top3" / "index.html").exists())
            self.assertIn("Top 3 Normal-Start Replays", (root / "replays" / "top3" / "index.html").read_text())
            for item in payload["replays"]:
                self.assertTrue(Path(str(item["stable_json"])).exists())
                self.assertTrue(Path(str(item["stable_html"])).exists())

    def test_sync_playlist_skips_when_current_and_updates_on_new_record(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "replays" / "top3"
            _write_top_game(root, "low", score=100, seed=1)
            _write_top_game(root, "mid", score=200, seed=2)
            _write_top_game(root, "high", score=300, seed=3)

            first = sync_top_replay_playlist(root=root, out_dir=out_dir, limit=3)
            second = sync_top_replay_playlist(root=root, out_dir=out_dir, limit=3)

            self.assertTrue(first["synced"])
            self.assertFalse(second["synced"])
            self.assertTrue(playlist_is_current(out_dir=out_dir, top_replays=collect_global_top_replays(root, limit=3), limit=3))

            _write_top_game(root, "record", score=400, seed=4)
            third = sync_top_replay_playlist(root=root, out_dir=out_dir, limit=3)

            self.assertTrue(third["synced"])
            self.assertEqual([item["score"] for item in third["replays"]], [400, 300, 200])


if __name__ == "__main__":
    unittest.main()

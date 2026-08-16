import argparse
import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.extract_continuation_replays import run_from_args


def make_entry(seed: int, max_tile_excl_starter: int, score_delta: int) -> dict:
    replay = {
        "seed": seed,
        "start_case_id": f"case-{seed}",
        "source_seed": 100 + seed,
        "source_frame_index": 10 + seed,
        "final_score": 1000 + score_delta,
        "final_score_delta": score_delta,
        "final_max_tile_excl_starter": max_tile_excl_starter,
        "frames": [{"index": 0, "state": {"board": [[0] * 4 for _ in range(4)]}}],
    }
    return {
        "record": {
            "seed": seed,
            "start_case_id": f"case-{seed}",
            "source_seed": 100 + seed,
            "source_frame_index": 10 + seed,
            "score": 1000 + score_delta,
            "score_delta": score_delta,
            "moves_delta": 12,
            "max_tile": max_tile_excl_starter,
            "max_tile_excl_starter": max_tile_excl_starter,
        },
        "replay": replay,
    }


def make_entry_without_replay_metadata(seed: int) -> dict:
    entry = make_entry(seed, 3072, 10)
    for key in (
        "start_case_id",
        "source_replay",
        "source_seed",
        "source_frame_index",
        "final_score_delta",
        "final_max_tile_excl_starter",
    ):
        entry["replay"].pop(key, None)
    entry["record"]["source_replay"] = "source/replay.json"
    return entry


class ExtractContinuationReplaysTests(unittest.TestCase):
    def test_extracts_and_filters_checkpoint_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = root / "progress.json"
            progress.write_text(
                json.dumps(
                    {
                        "entries": {
                            "a": make_entry(1, 3072, 10),
                            "b": make_entry(2, 6144, 20),
                        }
                    }
                )
            )
            out_dir = root / "out"

            payload = run_from_args(
                argparse.Namespace(
                    progress_json=progress,
                    min_max_tile_excl_starter=6144,
                    max_max_tile_excl_starter=0,
                    no_html=True,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["source_entries"], 2)
            self.assertEqual(payload["summary"]["replays"], 1)
            self.assertEqual(payload["summary"]["reached_6144"], 1)
            self.assertEqual(payload["summary"]["skipped"]["filter"], 1)
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest[0]["seed"], 2)
            self.assertEqual(manifest[0]["max_tile_excl_starter"], 6144)
            self.assertTrue(Path(manifest[0]["json"]).exists())

    def test_preserves_record_metadata_when_replay_omits_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            progress = root / "progress.json"
            progress.write_text(json.dumps({"entries": {"a": make_entry_without_replay_metadata(3)}}))
            out_dir = root / "out"

            payload = run_from_args(
                argparse.Namespace(
                    progress_json=progress,
                    min_max_tile_excl_starter=0,
                    max_max_tile_excl_starter=0,
                    no_html=True,
                    out_dir=out_dir,
                )
            )

            manifest = json.loads((out_dir / "manifest.json").read_text())
            replay = json.loads(Path(manifest[0]["json"]).read_text())
            self.assertEqual(payload["summary"]["replays"], 1)
            self.assertEqual(replay["start_case_id"], "case-3")
            self.assertEqual(replay["source_replay"], "source/replay.json")
            self.assertEqual(replay["source_seed"], 103)
            self.assertEqual(replay["source_frame_index"], 13)
            self.assertEqual(replay["final_score_delta"], 10)


if __name__ == "__main__":
    unittest.main()

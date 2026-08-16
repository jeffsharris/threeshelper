import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from threes_rl.replay_retention_audit import (
    build_replay_retention_audit,
    write_replay_retention_audit,
)


def _write_replay(root: Path, rel_path: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    path.with_name("replay.html").write_text("<html></html>")


def _summary_item(run_rel: str, score: int, seed: int) -> dict[str, object]:
    replay_rel = f"{run_rel}/top_games/rank_01_score_{score}_seed_{seed}_starter_1536"
    return {
        "seed": seed,
        "starter_tile": 1536,
        "score": score,
        "score_minus_starter": score - 1536,
        "moves": seed,
        "max_tile": 1536,
        "max_tile_excl_starter": 768,
        "html": f"threes_rl/runs/{replay_rel}/replay.html",
        "json": f"threes_rl/runs/{replay_rel}/replay.json",
    }


def _write_summary(root: Path, run_rel: str, item: dict[str, object]) -> None:
    path = root / run_rel / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"games": 1, "top_games": [item]}))


class ReplayRetentionAuditTests(unittest.TestCase):
    def test_audit_protects_global_top_and_excludes_non_normal_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            high = _summary_item("eval_artifacts/high", 500, 5)
            low = _summary_item("eval_artifacts/low", 100, 1)
            continuation = _summary_item("continuations/from_3072", 999, 9)
            replay_start = _summary_item("td_replay_start", 800, 8)
            for item in (high, low, continuation, replay_start):
                _write_replay(root, str(Path(str(item["json"])).relative_to("threes_rl/runs")))
            _write_summary(root, "eval_artifacts/high", high)
            _write_summary(root, "eval_artifacts/low", low)
            _write_summary(root, "continuations/from_3072", continuation)
            _write_summary(root, "td_replay_start", replay_start)
            (root / "td_replay_start" / "config.json").write_text(json.dumps({"start_state_prob": 0.4}))

            payload = build_replay_retention_audit(root, global_top_limit=1, preview_limit=10)

        self.assertEqual([item["score"] for item in payload["protected_global_top_replays"]], [500])
        self.assertEqual(payload["counts"]["top_game_entries_dashboard_eligible"], 2)
        self.assertEqual(payload["counts"]["top_game_entries_excluded_from_dashboard"], 2)
        self.assertEqual(payload["counts"]["non_global_top_game_entries"], 1)
        self.assertEqual(payload["potential_prune"]["preview"][0]["score"], 100)
        self.assertEqual(payload["retain_by_default"]["categories"]["continuations"], 1)

    def test_write_audit_is_non_destructive(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = _summary_item("eval_artifacts/high", 500, 5)
            _write_replay(root, str(Path(str(item["json"])).relative_to("threes_rl/runs")))
            _write_summary(root, "eval_artifacts/high", item)
            replay_path = root / Path(str(item["json"])).relative_to("threes_rl/runs")
            out_path = root / "dashboard" / "replay_retention_audit.json"

            payload = write_replay_retention_audit(out_path, root=root, global_top_limit=3)

            self.assertTrue(replay_path.exists())
            self.assertTrue(replay_path.with_name("replay.html").exists())
            self.assertTrue(out_path.exists())
            self.assertEqual(payload["mode"], "dry_run")


if __name__ == "__main__":
    unittest.main()

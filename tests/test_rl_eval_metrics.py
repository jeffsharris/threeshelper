import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from threes_rl.baselines import RandomPolicy
from threes_rl.compare_eval import summarize as summarize_eval_comparison
from threes_rl.eval import (
    EvalJob,
    GameResult,
    classify_death,
    iter_eval_job_outputs,
    max_tile_excluding_initial_starter,
    parse_starter_values,
    replay_key,
    run_game_with_optional_replay,
    selected_death_forensics_results,
    starter_baseline_score,
    summarize,
    write_death_forensics,
)
from threes_rl.run_artifacts import write_milestone_replays, write_pre_milestone_failure_replays, write_top_replays


class EvalMetricTests(unittest.TestCase):
    def test_starter_baseline_score(self):
        self.assertEqual(starter_baseline_score(None), 0)
        self.assertEqual(starter_baseline_score(1536), 59049)

    def test_parse_starter_values_accepts_curriculum(self):
        self.assertEqual(parse_starter_values("none,96,384,1536"), [None, 96, 384, 1536])

    def test_summarize_reports_by_starter_for_mixed_eval(self):
        summary = summarize(
            [
                GameResult(
                    seed=1,
                    starter_tile=None,
                    score=100,
                    score_minus_starter=100,
                    moves=10,
                    max_tile=96,
                    max_tile_excl_starter=96,
                    terminal_tile=False,
                ),
                GameResult(
                    seed=1,
                    starter_tile=96,
                    score=200,
                    score_minus_starter=119,
                    moves=12,
                    max_tile=192,
                    max_tile_excl_starter=192,
                    terminal_tile=False,
                ),
            ]
        )

        self.assertIn("by_starter", summary)
        by_starter = summary["by_starter"]
        self.assertEqual(by_starter["none"]["games"], 1)
        self.assertEqual(by_starter["96"]["mean_score_minus_starter"], 119)

    def test_selected_death_forensics_includes_worst_and_median(self):
        results = [
            GameResult(seed=1, score=110, score_minus_starter=10, moves=10, max_tile=96, max_tile_excl_starter=96, terminal_tile=False),
            GameResult(seed=2, score=120, score_minus_starter=20, moves=10, max_tile=96, max_tile_excl_starter=96, terminal_tile=False),
            GameResult(seed=3, score=130, score_minus_starter=30, moves=10, max_tile=96, max_tile_excl_starter=96, terminal_tile=False),
            GameResult(seed=4, score=140, score_minus_starter=40, moves=10, max_tile=96, max_tile_excl_starter=96, terminal_tile=False),
            GameResult(seed=5, score=150, score_minus_starter=50, moves=10, max_tile=96, max_tile_excl_starter=96, terminal_tile=False),
            GameResult(seed=6, score=160, score_minus_starter=60, moves=10, max_tile=96, max_tile_excl_starter=96, terminal_tile=False),
        ]

        selected = selected_death_forensics_results(results, worst_n=2)
        roles_by_seed = {result.seed: roles for roles, result in selected}

        self.assertIn("worst_1", roles_by_seed[1])
        self.assertIn("worst_2", roles_by_seed[2])
        self.assertIn("median", roles_by_seed[3])

    def test_classify_death_detects_bonus_clog_and_bag_starvation(self):
        final_state = {
            "board": [[1536, 384, 96, 48], [24, 12, 6, 3], [2, 1, 3, 6], [12, 24, 48, 96]],
            "game_over": True,
            "legal_actions": [],
            "preview": {"kind": "bonus", "label": "large_candidates", "candidates": [6, 12, 24]},
            "tile_cycle": {"small_counts": {"red": 0, "blue": 1, "gray": 0}},
        }

        classification = classify_death(final_state, [], 1536)

        self.assertIn("bonus_clog", classification["labels"])
        self.assertIn("bag_starvation", classification["labels"])

    def test_max_tile_excluding_initial_starter_ignores_free_corner_tile(self):
        board = np.asarray(
            [
                [1536, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        self.assertEqual(max_tile_excluding_initial_starter(board, 1536), 768)

    def test_max_tile_excluding_initial_starter_ignores_moved_free_tile(self):
        board = np.asarray(
            [
                [0, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 1536, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        self.assertEqual(max_tile_excluding_initial_starter(board, 1536), 768)

    def test_max_tile_excluding_initial_starter_counts_extra_starter_value_tile(self):
        board = np.asarray(
            [
                [0, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 1536, 0],
                [0, 0, 0, 1536],
            ],
            dtype=np.int32,
        )
        self.assertEqual(max_tile_excluding_initial_starter(board, 1536), 1536)

    def test_max_tile_excluding_initial_starter_counts_growth_in_corner(self):
        board = np.asarray(
            [
                [3072, 384, 0, 0],
                [0, 768, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        self.assertEqual(max_tile_excluding_initial_starter(board, 1536), 3072)

    def test_run_game_can_capture_replay_once(self):
        result, replay = run_game_with_optional_replay(
            RandomPolicy(),
            policy_name="random",
            seed=7,
            starter_tile=1536,
            max_moves=5,
            capture_replay=True,
        )

        self.assertIsNotNone(replay)
        assert replay is not None
        self.assertEqual(replay["final_score"], result.score)
        self.assertEqual(replay["final_moves"], result.moves)
        self.assertEqual(len(replay["frames"]), result.moves + 1)

    def test_parallel_eval_jobs_capture_replays(self):
        outputs = list(
            iter_eval_job_outputs(
                policy=RandomPolicy(),
                policy_name="random",
                eval_jobs=[
                    EvalJob(index=0, seed=7, starter_tile=1536),
                    EvalJob(index=1, seed=8, starter_tile=1536),
                ],
                max_moves=5,
                capture_replay=True,
                jobs=2,
            )
        )

        by_index = {output.index: output for output in outputs}

        self.assertEqual(set(by_index), {0, 1})
        self.assertEqual(by_index[0].result.seed, 7)
        self.assertEqual(by_index[1].result.seed, 8)
        self.assertIsNotNone(by_index[0].replay)
        assert by_index[0].replay is not None
        self.assertEqual(by_index[0].replay["final_score"], by_index[0].result.score)

    def test_write_top_replays_reuses_captured_replay(self):
        result, replay = run_game_with_optional_replay(
            RandomPolicy(),
            policy_name="random",
            seed=7,
            starter_tile=1536,
            max_moves=5,
            capture_replay=True,
        )
        assert replay is not None

        class FailingPolicy:
            def __call__(self, *_args):
                raise AssertionError("write_top_replays should not re-run captured policies")

        with TemporaryDirectory() as tmp:
            manifest = write_top_replays(
                run_dir=Path(tmp),
                results=[
                    GameResult(
                        seed=result.seed,
                        score=result.score,
                        score_minus_starter=result.score_minus_starter,
                        moves=result.moves,
                        max_tile=result.max_tile,
                        max_tile_excl_starter=result.max_tile_excl_starter,
                        terminal_tile=result.terminal_tile,
                    )
                ],
                policy=FailingPolicy(),
                policy_name="random",
                starter_tile=1536,
                max_moves=5,
                top_n=1,
                replays_by_seed={result.seed: replay},
            )

        self.assertEqual(manifest[0]["seed"], result.seed)

    def test_write_milestone_replays_reuses_captured_replay(self):
        result, replay = run_game_with_optional_replay(
            RandomPolicy(),
            policy_name="random",
            seed=7,
            starter_tile=1536,
            max_moves=5,
            capture_replay=True,
        )
        assert replay is not None

        class FailingPolicy:
            def __call__(self, *_args):
                raise AssertionError("write_milestone_replays should not re-run captured policies")

        with TemporaryDirectory() as tmp:
            manifest = write_milestone_replays(
                run_dir=Path(tmp),
                results=[
                    GameResult(
                        seed=result.seed,
                        starter_tile=result.starter_tile,
                        score=result.score,
                        score_minus_starter=result.score_minus_starter,
                        moves=result.moves,
                        max_tile=result.max_tile,
                        max_tile_excl_starter=result.max_tile_excl_starter,
                        terminal_tile=result.terminal_tile,
                    )
                ],
                policy=FailingPolicy(),
                policy_name="random",
                starter_tile=1536,
                max_moves=5,
                threshold=0,
                replays_by_seed={replay_key(result.seed, result.starter_tile): replay},
            )
            manifest_path = Path(tmp) / "milestone_games" / "ge_0" / "manifest.json"
            manifest_exists = manifest_path.exists()

        self.assertEqual(manifest["qualified_games"], 1)
        self.assertEqual(manifest["replays"][0]["seed"], result.seed)
        self.assertTrue(manifest_exists)

    def test_write_pre_milestone_failure_replays_reuses_captured_replay_and_caps(self):
        _result, replay = run_game_with_optional_replay(
            RandomPolicy(),
            policy_name="random",
            seed=7,
            starter_tile=1536,
            max_moves=5,
            capture_replay=True,
        )
        assert replay is not None

        class FailingPolicy:
            def __call__(self, *_args):
                raise AssertionError("write_pre_milestone_failure_replays should not re-run captured policies")

        results = [
            GameResult(
                seed=7,
                starter_tile=1536,
                score=1000,
                score_minus_starter=100,
                moves=50,
                max_tile=1536,
                max_tile_excl_starter=1536,
                terminal_tile=False,
            ),
            GameResult(
                seed=8,
                starter_tile=1536,
                score=2000,
                score_minus_starter=200,
                moves=60,
                max_tile=1536,
                max_tile_excl_starter=1536,
                terminal_tile=False,
            ),
            GameResult(
                seed=9,
                starter_tile=1536,
                score=3000,
                score_minus_starter=300,
                moves=70,
                max_tile=3072,
                max_tile_excl_starter=3072,
                terminal_tile=False,
            ),
            GameResult(
                seed=10,
                starter_tile=1536,
                score=4000,
                score_minus_starter=400,
                moves=80,
                max_tile=768,
                max_tile_excl_starter=768,
                terminal_tile=False,
            ),
        ]

        with TemporaryDirectory() as tmp:
            manifest = write_pre_milestone_failure_replays(
                run_dir=Path(tmp),
                results=results,
                policy=FailingPolicy(),
                policy_name="random",
                starter_tile=1536,
                max_moves=5,
                min_tile=1536,
                threshold=3072,
                max_games=1,
                replays_by_seed={
                    replay_key(7, 1536): replay,
                    replay_key(8, 1536): replay,
                },
            )
            manifest_path = Path(tmp) / "diagnostic_games" / "pre_3072_min_1536" / "manifest.json"
            manifest_exists = manifest_path.exists()

        self.assertEqual(manifest["qualified_games"], 2)
        self.assertEqual(manifest["retained_games"], 1)
        self.assertEqual(manifest["replays"][0]["seed"], 8)
        self.assertTrue(manifest_exists)

    def test_write_death_forensics_reuses_captured_replay(self):
        result, replay = run_game_with_optional_replay(
            RandomPolicy(),
            policy_name="random",
            seed=7,
            starter_tile=1536,
            max_moves=5,
            capture_replay=True,
        )
        assert replay is not None

        class FailingPolicy:
            def __call__(self, *_args):
                raise AssertionError("write_death_forensics should reuse captured replays")

        with TemporaryDirectory() as tmp:
            manifest = write_death_forensics(
                run_dir=Path(tmp),
                results=[result],
                policy=FailingPolicy(),
                policy_name="random",
                max_moves=5,
                replays_by_seed={replay_key(result.seed, result.starter_tile): replay},
            )
            payload = (Path(tmp) / "death_forensics.json").read_text()

        self.assertEqual(manifest["cases"], 1)
        self.assertIn("final_board", payload)
        self.assertIn("death_forensics.html", manifest["html"])

    def test_compare_eval_summarizes_paired_csvs(self):
        with TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.csv"
            baseline = Path(tmp) / "baseline.csv"
            candidate.write_text("seed,score_minus_starter\n1,10\n2,20\n3,30\n")
            baseline.write_text("seed,score_minus_starter\n1,5\n2,25\n4,100\n")

            summary = summarize_eval_comparison([candidate], [baseline], "score_minus_starter", 2)

        self.assertEqual(summary["paired_seeds"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["paired_mean_diff"], 0)


if __name__ == "__main__":
    unittest.main()

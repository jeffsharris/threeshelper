import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from threes_rl.dashboard import (
    Point,
    collect_frontier_progress,
    collect_global_top_replays,
    collect_human_inbox_status,
    collect_points,
    collect_replay_retention_status,
    collect_top_replay_playlist_status,
    dashboard_payload,
    score_trends_payload,
    write_html,
    write_score_trends,
    write_score_trends_html,
)


class DashboardTests(unittest.TestCase):
    def test_dashboard_payload_picks_best_and_latest(self):
        points = [
            Point(
                label="early",
                path="runs/early",
                kind="eval",
                high_score=100,
                high_score_minus_starter=50,
                mean_score_minus_starter=20,
                median_score_minus_starter=10,
                p3072=0.0,
                p6144=0.0,
                games=10,
                mtime=1,
            ),
            Point(
                label="late",
                path="runs/late",
                kind="eval",
                high_score=90,
                high_score_minus_starter=60,
                mean_score_minus_starter=30,
                median_score_minus_starter=20,
                p3072=0.1,
                p6144=0.0,
                games=20,
                mtime=2,
            ),
        ]

        payload = dashboard_payload(points)

        self.assertEqual(payload["best"]["label"], "early")
        self.assertEqual(payload["latest"]["label"], "late")
        self.assertEqual(payload["best_high_score"], 100)
        self.assertEqual(payload["best_high_score_minus_starter"], 50)
        self.assertEqual(payload["latest_high_score"], 90)
        self.assertEqual(payload["latest_mean_score_minus_starter"], 30)
        self.assertEqual(payload["latest_median_score_minus_starter"], 20)
        self.assertEqual(payload["latest_p3072"], 0.1)
        self.assertEqual(payload["global_top_replays"], [])
        self.assertEqual(payload["global_top_scores"], [])
        self.assertEqual(payload["global_top_replay_limit"], 3)

    def test_dashboard_payload_caps_global_top_replays(self):
        payload = dashboard_payload(
            [
                Point(
                    label="run",
                    path="runs/run",
                    kind="eval",
                    high_score=100,
                    high_score_minus_starter=50,
                    mean_score_minus_starter=20,
                    median_score_minus_starter=10,
                    p3072=0.0,
                    p6144=0.0,
                    games=10,
                    mtime=1,
                )
            ],
            top_replays=[
                {"score": 4},
                {"score": 3},
                {"score": 2},
                {"score": 1},
            ],
        )

        self.assertEqual([item["score"] for item in payload["global_top_replays"]], [4, 3, 2])
        self.assertEqual(payload["global_top_scores"], [4, 3, 2])
        self.assertEqual(payload["global_top_replay_limit"], 3)

    def test_dashboard_payload_includes_ops_statuses(self):
        payload = dashboard_payload(
            [],
            human_inbox={"status": "waiting_for_human_data"},
            replay_retention={"status": "ok"},
            top_replay_playlist={"replays": [{"score": 300}, {"score": 200}, {"score": 100}]},
        )

        self.assertEqual(payload["human_inbox"]["status"], "waiting_for_human_data")
        self.assertEqual(payload["replay_retention"]["status"], "ok")
        self.assertEqual(payload["top_replay_playlist"]["scores"], [300, 200, 100])
        self.assertEqual(payload["top_replay_playlist_scores"], [300, 200, 100])
        self.assertIsNone(payload["best_high_score"])

    def test_collect_human_inbox_status(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "human_diagnostics" / "human_diagnostics_batch.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "waiting_for_human_data",
                        "mode": "dry_run",
                        "html": "threes_rl/runs/human_diagnostics/human_diagnostics_batch.html",
                        "totals": {
                            "sessions": 2,
                            "pending_sessions": 1,
                            "current_sessions": 1,
                            "processed_sessions": 0,
                            "games_imported": 3,
                            "high_score": 12345,
                            "highest_max_tile_excl_starter": 3072,
                            "games_reaching_nonstarter_1536": 2,
                            "games_reaching_3072": 1,
                        },
                        "target_intake": {
                            "independent_games_reaching_nonstarter_1536": 5,
                            "independent_games_reaching_3072": 1,
                            "ready_for_human_root_labeling": False,
                        },
                    }
                )
            )

            status = collect_human_inbox_status(root)

        self.assertEqual(status["status"], "waiting_for_human_data")
        self.assertEqual(status["sessions"], 2)
        self.assertEqual(status["pending_sessions"], 1)
        self.assertEqual(status["games_imported"], 3)
        self.assertEqual(status["high_score"], 12345)
        self.assertEqual(status["highest_max_tile_excl_starter"], 3072)
        self.assertEqual(status["games_reaching_nonstarter_1536"], 2)
        self.assertEqual(status["games_reaching_3072"], 1)
        self.assertEqual(status["target_nonstarter_1536"], 5)
        self.assertEqual(status["target_3072"], 1)
        self.assertFalse(status["ready_for_human_root_labeling"])

    def test_collect_replay_retention_status(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "dashboard" / "replay_retention_audit.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "mode": "dry_run",
                        "global_top_limit": 3,
                        "protected_global_top_replays": [
                            {"score": 300},
                            {"score": 200},
                            {"score": 100},
                        ],
                        "counts": {
                            "missing_protected_global_top_json": 0,
                            "non_global_top_game_entries": 7,
                            "replay_dirs": 12,
                        },
                    }
                )
            )

            status = collect_replay_retention_status(root)

        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["protected_count"], 3)
        self.assertEqual(status["protected_scores"], [300, 200, 100])
        self.assertEqual(status["potential_prune_count"], 7)

    def test_collect_top_replay_playlist_status(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "replays" / "top3" / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "html": "threes_rl/runs/replays/top3/index.html",
                        "generated_at": "2026-07-08T15:00:00-0700",
                        "copied_count": 3,
                        "replays": [{"score": 300}, {"score": 200}, {"score": 100}],
                    }
                )
            )

            status = collect_top_replay_playlist_status(root)

        self.assertEqual(status["copied_count"], 3)
        self.assertEqual(status["scores"], [300, 200, 100])
        self.assertEqual(status["html"], "threes_rl/runs/replays/top3/index.html")

    def test_write_html_renders_ops_status_panel(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "dashboard.html"
            payload = dashboard_payload(
                [],
                human_inbox={
                    "status": "waiting_for_human_data",
                    "sessions": 0,
                    "games_imported": 0,
                    "games_reaching_nonstarter_1536": 0,
                    "games_reaching_3072": 0,
                },
                replay_retention={
                    "status": "ok",
                    "protected_count": 3,
                    "global_top_limit": 3,
                    "protected_scores": [300, 200, 100],
                    "potential_prune_count": 7,
                },
                top_replay_playlist={
                    "html": "threes_rl/runs/replays/top3/index.html",
                    "copied_count": 3,
                    "scores": [300, 200, 100],
                },
            )

            write_html(out, payload, refresh_seconds=0)
            html = out.read_text()

        self.assertIn("Research Inputs & Retention", html)
        self.assertIn("Human data inbox", html)
        self.assertIn("Replay retention", html)
        self.assertIn("non-starter 1536", html)
        self.assertIn("open playlist", html)

    def test_frontier_progress_is_kept_separate_from_full_game_record(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = root / "forensics" / "frontier_compare" / "local_chain_to_6144_test" / "summary.json"
            chain.parent.mkdir(parents=True)
            chain.write_text(
                json.dumps(
                    {
                        "rungs": {
                            "adjacent_1536_to_second_3072": {
                                "summary": {
                                    "target_hits": 9,
                                    "valid_rollouts": 10,
                                    "target_rate": 0.9,
                                    "cases_selected": 2,
                                    "ancestries_selected": 2,
                                    "horizon": 40,
                                }
                            },
                            "second_3072_to_6144": {
                                "summary": {
                                    "target_hits": 8,
                                    "valid_rollouts": 10,
                                    "target_rate": 0.8,
                                    "cases_selected": 2,
                                    "ancestries_selected": 2,
                                    "horizon": 40,
                                    "created_at": "2026-07-08T04:00:00-0700",
                                }
                            },
                        }
                    }
                )
            )
            barrier = (
                root
                / "forensics"
                / "frontier_compare"
                / "root1074_nearadj1536_chain_test"
                / "summary.json"
            )
            barrier.parent.mkdir(parents=True)
            barrier.write_text(
                json.dumps(
                    {
                        "created_at": "2026-07-08T05:00:00-0700",
                        "steps": [
                            {
                                "label": "duplicate -> near adjacent",
                                "hits": 1,
                                "rollouts": 20,
                                "rate": 0.05,
                                "cases": 2,
                                "positive_cases": 1,
                                "horizon": 40,
                            }
                        ],
                        "interpretation": "The geometry transition is still sparse.",
                    }
                )
            )

            frontier = collect_frontier_progress(root)
            payload = dashboard_payload(
                [
                    Point(
                        label="full-game",
                        path="runs/full-game",
                        kind="eval",
                        high_score=100,
                        high_score_minus_starter=50,
                        mean_score_minus_starter=20,
                        median_score_minus_starter=10,
                        p3072=0.0,
                        p6144=0.0,
                        games=10,
                        mtime=1,
                    )
                ],
                frontier_progress=frontier,
            )

        self.assertIsNotNone(frontier)
        self.assertEqual(frontier["highest_milestone"], 6144)
        self.assertEqual(len(frontier["transitions"]), 3)
        self.assertEqual(payload["best"]["high_score"], 100)
        self.assertEqual(payload["frontier_progress"]["status"], "diagnostic_only")

    def test_write_html_creates_dashboard_page(self):
        payload = dashboard_payload(
            [
                Point(
                    label="run",
                    path="runs/run",
                    kind="eval",
                    high_score=100,
                    high_score_minus_starter=50,
                    mean_score_minus_starter=20,
                    median_score_minus_starter=10,
                    p3072=0.0,
                    p6144=0.0,
                    games=10,
                    mtime=1,
                    annotation="New high-score record.",
                )
            ]
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.html"
            write_html(path, payload, refresh_seconds=0)

            html = path.read_text()

        self.assertIn("Threes RL Research Dashboard", html)
        self.assertIn("Global Top 3 Normal-Start Replays", html)
        self.assertIn("Frontier Transition Research", html)
        self.assertIn("Diagnostic only", html)
        self.assertIn("New high-score record.", html)
        self.assertIn("Best high score", html)
        self.assertIn('id="chartFreshness"', html)
        self.assertIn('id="chartEligibility"', html)
        self.assertIn("Latest research result", html)
        self.assertIn("score_trends.html", html)
        self.assertNotIn("Mean / Median Trend", html)
        self.assertNotIn("trendChart", html)
        self.assertIn("Run time", html)
        self.assertIn("<polyline", html)
        self.assertIn('class="hit"', html)
        self.assertIn('tabindex="0"', html)
        self.assertIn('aria-label=', html)
        self.assertIn('"pointerover"', html)
        self.assertIn('"focusin"', html)
        self.assertNotIn("Score Progress Timeline", html)

    def test_score_trends_payload_and_html_include_mean_median_series(self):
        points = [
            Point(
                label="run-a",
                path="runs/run-a",
                kind="eval",
                high_score=100,
                high_score_minus_starter=50,
                mean_score_minus_starter=20,
                median_score_minus_starter=10,
                p3072=0.0,
                p6144=0.0,
                games=10,
                mtime=1,
            ),
            Point(
                label="run-b",
                path="runs/run-b",
                kind="eval",
                high_score=200,
                high_score_minus_starter=150,
                mean_score_minus_starter=40,
                median_score_minus_starter=30,
                p3072=0.1,
                p6144=0.0,
                games=20,
                mtime=2,
            ),
        ]
        payload = score_trends_payload(points)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "score_trends.html"
            write_score_trends_html(path, payload)
            html = path.read_text()

        self.assertEqual(len(payload["points"]), 2)
        self.assertIn("High / Mean / Median Trend", html)
        self.assertIn("mean_score_minus_starter", html)
        self.assertIn("median_score_minus_starter", html)
        self.assertIn("metricsChart", html)

    def test_write_score_trends_writes_json_and_html(self):
        points = [
            Point(
                label="run",
                path="runs/run",
                kind="eval",
                high_score=100,
                high_score_minus_starter=50,
                mean_score_minus_starter=20,
                median_score_minus_starter=10,
                p3072=0.0,
                p6144=0.0,
                games=10,
                mtime=1,
            )
        ]
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            payload = write_score_trends(out_dir, points)

            self.assertTrue((out_dir / "score_trends.json").exists())
            self.assertTrue((out_dir / "score_trends.html").exists())
            self.assertEqual(payload["points"][0]["label"], "run")

    def test_collect_points_prefers_created_at_for_time_axis(self):
        with TemporaryDirectory() as tmp:
            summary = Path(tmp) / "eval_artifacts" / "run" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "created_at": "2026-07-05T10:30:00-0700",
                        "games": 1,
                        "high_score": 100,
                        "high_score_minus_starter": 50,
                    }
                )
            )

            points = collect_points(Path(tmp))

        expected = datetime.fromisoformat("2026-07-05T10:30:00-07:00").timestamp()
        self.assertEqual(points[0].created_at, "2026-07-05T10:30:00-0700")
        self.assertAlmostEqual(points[0].mtime, expected)

    def test_collect_points_ignores_ephemeral_test_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_summary = root / "td_real_experiment" / "summary.json"
            real_summary.parent.mkdir(parents=True)
            real_summary.write_text(json.dumps({"games": 1, "high_score": 100}))
            tmp_summary = root / "td_smoke_tmpabc123" / "summary.json"
            tmp_summary.parent.mkdir(parents=True)
            tmp_summary.write_text(json.dumps({"games": 1, "high_score": 999}))

            points = collect_points(root)

        self.assertEqual([point.label for point in points], ["td_real_experiment"])

    def test_collect_points_excludes_continuation_summaries(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_game = root / "eval_artifacts" / "full_game" / "summary.json"
            full_game.parent.mkdir(parents=True)
            full_game.write_text(json.dumps({"games": 1, "high_score": 100}))
            continuation = root / "continuations" / "from_3072" / "summary.json"
            continuation.parent.mkdir(parents=True)
            continuation.write_text(json.dumps({"games": 1, "high_score": 999}))

            points = collect_points(root)

        self.assertEqual([point.label for point in points], ["full_game"])

    def test_collect_points_excludes_replay_start_training_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_game = root / "td_normal" / "summary.json"
            full_game.parent.mkdir(parents=True)
            full_game.write_text(json.dumps({"games": 1, "high_score": 100}))
            replay_start = root / "td_replay_start" / "summary.json"
            replay_start.parent.mkdir(parents=True)
            replay_start.write_text(json.dumps({"games": 1, "high_score": 999}))
            (replay_start.parent / "config.json").write_text(
                json.dumps(
                    {
                        "start_state_prob": 1.0,
                        "start_state_replays": ["high_score_replay.json"],
                    }
                )
            )

            points = collect_points(root)

        self.assertEqual([point.label for point in points], ["td_normal"])

    def test_collect_points_excludes_replay_start_progress_rows(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay_start = root / "td_replay_start" / "progress.csv"
            replay_start.parent.mkdir(parents=True)
            replay_start.write_text(
                "created_at,games,high_score,high_score_minus_starter\n"
                "2026-07-06T10:00:00-0700,1,999,900\n"
            )
            (replay_start.parent / "config.json").write_text(
                json.dumps({"start_state_prob": 0.6, "start_state_replays": ["replay.json"]})
            )

            points = collect_points(root)

        self.assertEqual(points, [])

    def test_failed_confirmation_is_visible_but_not_record_eligible(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            incumbent = root / "eval_artifacts" / "incumbent" / "summary.json"
            incumbent.parent.mkdir(parents=True)
            incumbent.write_text(json.dumps({"games": 1, "high_score": 200}))
            candidate = root / "eval_artifacts" / "candidate_confirmation" / "summary.json"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(json.dumps({"games": 1, "high_score": 999, "blocks": ["C"]}))
            (candidate.parent / "confirmation_lock.json").write_text(
                json.dumps({"decision": "CONFIRMATION_FAILED_NO_PROMOTION_HOLD_FOR_REVIEW"})
            )

            points = collect_points(root)
            payload = dashboard_payload(points)

        self.assertEqual([point.label for point in points], ["incumbent", "candidate_confirmation"])
        self.assertFalse(points[1].record_eligible)
        self.assertEqual(payload["best"]["label"], "incumbent")
        self.assertEqual(payload["best_high_score"], 200)
        self.assertNotIn("New high-score record.", points[1].annotation or "")

    def test_development_candidate_replay_cannot_enter_global_top(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            incumbent = root / "eval_artifacts" / "incumbent" / "summary.json"
            incumbent.parent.mkdir(parents=True)
            incumbent.write_text(
                json.dumps({"high_score": 200, "top_games": [{"seed": 1, "score": 200}]})
            )
            candidate = root / "eval_artifacts" / "candidate_d2" / "summary.json"
            candidate.parent.mkdir(parents=True)
            candidate.write_text(
                json.dumps(
                    {
                        "blocks": ["D2"],
                        "high_score": 999,
                        "top_games": [{"seed": 2, "score": 999}],
                    }
                )
            )

            top = collect_global_top_replays(root, limit=3)

        self.assertEqual([item["score"] for item in top], [200])

    def test_collect_global_top_replays_sorts_and_excludes_continuations(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            low = root / "eval_artifacts" / "low" / "summary.json"
            low.parent.mkdir(parents=True)
            low.write_text(
                json.dumps(
                    {
                        "games": 1,
                        "high_score": 100,
                        "top_games": [
                            {
                                "seed": 1,
                                "score": 100,
                                "score_minus_starter": 50,
                                "moves": 10,
                                "html": "threes_rl/runs/eval_artifacts/low/top_games/low/replay.html",
                                "json": "threes_rl/runs/eval_artifacts/low/top_games/low/replay.json",
                            }
                        ],
                    }
                )
            )
            high = root / "eval_artifacts" / "high" / "summary.json"
            high.parent.mkdir(parents=True)
            high.write_text(
                json.dumps(
                    {
                        "games": 1,
                        "high_score": 200,
                        "top_games": [
                            {
                                "seed": 2,
                                "score": 200,
                                "score_minus_starter": 150,
                                "moves": 20,
                                "html": "threes_rl/runs/eval_artifacts/high/top_games/high/replay.html",
                                "json": "threes_rl/runs/eval_artifacts/high/top_games/high/replay.json",
                            }
                        ],
                    }
                )
            )
            duplicate = root / "eval_artifacts" / "duplicate_high" / "summary.json"
            duplicate.parent.mkdir(parents=True)
            duplicate.write_text(
                json.dumps(
                    {
                        "games": 1,
                        "high_score": 200,
                        "top_games": [
                            {
                                "seed": 2,
                                "score": 200,
                                "score_minus_starter": 150,
                                "moves": 20,
                                "html": "threes_rl/runs/eval_artifacts/duplicate_high/top_games/high/replay.html",
                                "json": "threes_rl/runs/eval_artifacts/duplicate_high/top_games/high/replay.json",
                            }
                        ],
                    }
                )
            )
            continuation = root / "continuations" / "from_3072" / "summary.json"
            continuation.parent.mkdir(parents=True)
            continuation.write_text(
                json.dumps(
                    {
                        "games": 1,
                        "high_score": 999,
                        "top_games": [{"seed": 3, "score": 999}],
                    }
                )
            )

            top = collect_global_top_replays(root, limit=2)

        self.assertEqual([item["score"] for item in top], [200, 100])
        self.assertIn(top[0]["run"], {"high", "duplicate_high"})
        self.assertEqual(top[1]["run"], "low")


if __name__ == "__main__":
    unittest.main()

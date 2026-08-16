import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.sim import DIRECTION_NAMES, SimState, ThreesSim, preview_from_label, simulate_base_move
from threes_rl.swing_label import (
    candidate_scope_key,
    collect_anchor_risk_states,
    collect_anchor_risk_states_from_replays,
    collect_anchor_risk_states_from_state_records,
    collect_geometry_risk_states_from_replays,
    collect_geometry_risk_states_from_state_records,
    collect_support_chain_risk_states_from_replays,
    collect_support_chain_risk_states_from_state_records,
    collect_swing_states,
    collect_swing_states_from_replays,
    collect_top_two_states_from_state_records,
    label_corpus,
    label_corpus_resumable_parallel,
    label_corpus_resumable,
    load_action_value_cache,
    load_samples_json,
    normalized_margin,
    parse_filter_values,
    rollout_action,
    rollout_action_metrics,
    sample_scope_key,
    summarize_labeled,
    support_chain_features,
    write_action_value_cache,
)


class FirstLegalNearTiePolicy:
    def __call__(self, state, sim, rng):
        return int(sim.legal_actions(state)[0])

    def _root_depth(self, state):
        return 1

    def _action_value(self, state, sim, action, depth):
        return 100.0 - 0.01 * int(action)


class LastLegalPolicy:
    def __call__(self, state, sim, rng):
        return int(sim.legal_actions(state)[-1])


class ExplodingPolicy:
    def __call__(self, state, sim, rng):
        raise AssertionError("resume should not recompute completed repeats")


class AnchorRiskPolicy:
    def __call__(self, state, sim, rng):
        legal = sim.legal_actions(state)
        right = DIRECTION_NAMES.index("right")
        return right if right in legal else int(legal[0])

    def _root_depth(self, state):
        return 1

    def _action_value(self, state, sim, action, depth):
        values = {
            "right": 100.0,
            "up": 99.9,
            "left": 75.0,
            "down": 50.0,
        }
        return values[DIRECTION_NAMES[int(action)]]


class CountingAnchorRiskPolicy(AnchorRiskPolicy):
    def __init__(self):
        self.action_value_calls = 0

    def _action_value(self, state, sim, action, depth):
        self.action_value_calls += 1
        return super()._action_value(state, sim, action, depth)


class GeometryRiskPolicy:
    def __call__(self, state, sim, rng):
        legal = sim.legal_actions(state)
        right = DIRECTION_NAMES.index("right")
        return right if right in legal else int(legal[0])

    def _root_depth(self, state):
        return 1

    def _action_value(self, state, sim, action, depth):
        values = {
            "right": 100.0,
            "left": 99.5,
            "down": 80.0,
            "up": 40.0,
        }
        return values[DIRECTION_NAMES[int(action)]]


class SupportChainRiskPolicy:
    def __call__(self, state, sim, rng):
        legal = sim.legal_actions(state)
        right = DIRECTION_NAMES.index("right")
        return right if right in legal else int(legal[0])

    def _root_depth(self, state):
        return 1

    def _action_value(self, state, sim, action, depth):
        values = {
            "right": 100.0,
            "down": 99.5,
            "up": 80.0,
            "left": 70.0,
        }
        return values[DIRECTION_NAMES[int(action)]]


def simple_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1, 2, 0, 0],
                [3, 3, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3,
        move_count=0,
        game_over=False,
    )


def anchor_risk_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [1536, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=1536,
        move_count=0,
        game_over=False,
    )


def geometry_risk_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [0, 3072, 1536, 0],
                [0, 3, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=0,
        game_over=False,
    )


def support_chain_risk_state() -> SimState:
    return SimState(
        board=np.asarray(
            [
                [0, 768, 0, 0],
                [3072, 1536, 0, 1536],
                [0, 0, 0, 0],
                [1536, 768, 0, 0],
            ],
            dtype=np.int32,
        ),
        preview=preview_from_label("gray"),
        small_counts={"red": 4, "blue": 4, "gray": 4},
        small_pos=0,
        small_seen_total=0,
        span_small_pos=0,
        large_pending=False,
        max_tile=3072,
        move_count=0,
        game_over=False,
    )


class SwingLabelTests(unittest.TestCase):
    def test_normalized_margin_uses_value_scale(self):
        margin, normalized = normalized_margin(
            [
                {"value": 100.0},
                {"value": 99.0},
            ]
        )

        self.assertEqual(margin, 1.0)
        self.assertAlmostEqual(normalized, 0.01)

    def test_rollout_action_is_deterministic_for_same_seeds(self):
        policy = FirstLegalNearTiePolicy()
        state = simple_state()

        left = rollout_action(
            state=state,
            first_action=DIRECTION_NAMES.index("left"),
            policy=policy,
            starter_tile=None,
            horizons=(2, 4),
            sim_seed=123,
            policy_seed=456,
        )
        again = rollout_action(
            state=state,
            first_action=DIRECTION_NAMES.index("left"),
            policy=policy,
            starter_tile=None,
            horizons=(2, 4),
            sim_seed=123,
            policy_seed=456,
        )

        self.assertEqual(left, again)
        self.assertEqual(sorted(left), [2, 4])

    def test_rollout_action_metrics_tracks_target_promotion(self):
        policy = FirstLegalNearTiePolicy()
        state = simple_state()

        metrics = rollout_action_metrics(
            state=state,
            first_action=DIRECTION_NAMES.index("left"),
            policy=policy,
            starter_tile=None,
            horizons=(1, 2),
            sim_seed=123,
            policy_seed=456,
            target_tile=3,
        )

        self.assertEqual(sorted(metrics), [1, 2])
        self.assertIn("score_delta", metrics[1])
        self.assertIn("max_tile_excl_starter", metrics[1])
        self.assertTrue(metrics[1]["target_reached"])

    def test_collect_and_label_small_corpus(self):
        corpus = collect_swing_states(
            base_policy=FirstLegalNearTiePolicy(),
            base_policy_spec="base",
            comparison_policies=[("last", LastLegalPolicy())],
            seeds=[7],
            starter_tile=1536,
            max_moves=5,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
        )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertNotEqual(sample["base_action"], sample["comparison_action"])
        self.assertEqual(len(sample["top_two_actions"]), 2)

        labels = label_corpus(
            corpus["samples"],
            policy=FirstLegalNearTiePolicy(),
            repeats=2,
            horizons=(2, 4),
            label_seed=20260706,
            stability_threshold=0.7,
        )
        summary = summarize_labeled(labels, corpus["scan_stats"])

        self.assertEqual(summary["labels"], 1)
        self.assertIn("horizon_consistent_label_rate", summary)
        self.assertIn("stable_label_rate", summary)
        self.assertIn("oracle_positive_regrets", summary)

    def test_transition_label_includes_promotion_rates(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=None)
        sample = {
            "id": "promotion-sample",
            "sample_mode": "top-two",
            "seed": 1,
            "starter_tile": None,
            "target_tile": 3,
            "base_action": "left",
            "comparison_action": "right",
            "top_two_actions": ["left", "right"],
            "features": {"stratum": "early_lt384/low_corner_risk"},
            "state": state_payload(state, sim),
        }

        labels = label_corpus(
            [sample],
            policy=FirstLegalNearTiePolicy(),
            repeats=2,
            horizons=(1, 2),
            label_seed=20260706,
            stability_threshold=0.7,
        )
        label = labels[0]["label"]
        summary = summarize_labeled(labels, {"accepted_samples": 1})

        self.assertEqual(label["target_tile"], 3)
        self.assertIn("promotion_rate_at_max_horizon", label)
        self.assertIn("promotion_winner_at_max_horizon", label)
        self.assertIn("promotion_rate", label["horizon_results"][0])
        self.assertEqual(summary["promotion_labels"], 1)
        self.assertEqual(summary["promotion_hit_labels"], 1)

    def test_resumable_labels_reuse_completed_repeats_and_extend(self):
        corpus = collect_swing_states(
            base_policy=FirstLegalNearTiePolicy(),
            base_policy_spec="base",
            comparison_policies=[("last", LastLegalPolicy())],
            seeds=[7],
            starter_tile=1536,
            max_moves=5,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "label_progress.json"
            labels = label_corpus_resumable(
                corpus["samples"],
                policy=FirstLegalNearTiePolicy(),
                policy_spec="base",
                repeats=1,
                horizons=(2,),
                label_seed=20260706,
                stability_threshold=0.7,
                progress_path=progress_path,
                repeat_chunk_size=1,
            )
            resumed = label_corpus_resumable(
                corpus["samples"],
                policy=ExplodingPolicy(),
                policy_spec="base",
                repeats=1,
                horizons=(2,),
                label_seed=20260706,
                stability_threshold=0.7,
                progress_path=progress_path,
                repeat_chunk_size=1,
            )
            extended = label_corpus_resumable(
                corpus["samples"],
                policy=FirstLegalNearTiePolicy(),
                policy_spec="base",
                repeats=2,
                horizons=(2,),
                label_seed=20260706,
                stability_threshold=0.7,
                progress_path=progress_path,
                repeat_chunk_size=1,
            )
            payload = json.loads(progress_path.read_text())

        self.assertEqual(resumed[0]["label"]["by_action"], labels[0]["label"]["by_action"])
        self.assertEqual(extended[0]["label"]["repeats"], 2)
        entry = next(iter(payload["samples"].values()))
        self.assertEqual(entry["completed_repeats"], [0, 1])

    def test_resumable_parallel_reuses_completed_progress_without_workers(self):
        corpus = collect_swing_states(
            base_policy=FirstLegalNearTiePolicy(),
            base_policy_spec="base",
            comparison_policies=[("last", LastLegalPolicy())],
            seeds=[7],
            starter_tile=1536,
            max_moves=5,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "label_progress.json"
            labels = label_corpus_resumable(
                corpus["samples"],
                policy=FirstLegalNearTiePolicy(),
                policy_spec="base",
                repeats=1,
                horizons=(2,),
                label_seed=20260706,
                stability_threshold=0.7,
                progress_path=progress_path,
                repeat_chunk_size=1,
            )
            resumed = label_corpus_resumable_parallel(
                corpus["samples"],
                policy_spec="base",
                repeats=1,
                horizons=(2,),
                label_seed=20260706,
                stability_threshold=0.7,
                progress_path=progress_path,
                workers=2,
                repeat_chunk_size=1,
            )

        self.assertEqual(resumed[0]["label"]["by_action"], labels[0]["label"]["by_action"])

    def test_collect_can_filter_phase_buckets(self):
        corpus = collect_swing_states(
            base_policy=FirstLegalNearTiePolicy(),
            base_policy_spec="base",
            comparison_policies=[("last", LastLegalPolicy())],
            seeds=[7],
            starter_tile=1536,
            max_moves=5,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
            phase_filter={"late_1536"},
        )

        self.assertEqual(corpus["samples"], [])
        self.assertGreater(corpus["scan_stats"]["rejected"].get("phase_filter", 0), 0)

    def test_collect_anchor_risk_states_uses_best_safe_challenger(self):
        corpus = collect_anchor_risk_states(
            base_policy=AnchorRiskPolicy(),
            base_policy_spec="anchor-risk-test",
            seeds=[1],
            starter_tile=1536,
            max_moves=10,
            margin_threshold=0.002,
            max_samples=1,
            max_per_stratum=1,
            first_per="seed-stratum",
        )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["sample_mode"], "anchor-risk")
        self.assertEqual(sample["base_action"], "right")
        self.assertEqual(sample["comparison_action"], "up")
        self.assertEqual(sample["top_two_actions"], ["right", "up"])
        self.assertLessEqual(sample["normalized_margin"], 0.002)
        self.assertEqual(sample["anchor"]["best_safe_action"], "up")
        self.assertEqual(corpus["scan_stats"]["sample_mode"], "anchor-risk")

    def test_collect_anchor_risk_states_from_replay_frames(self):
        state = anchor_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        replay = {
            "policy": "anchor-risk-test",
            "seed": 1,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": "right"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(replay))

            corpus = collect_anchor_risk_states_from_replays(
                base_policy=AnchorRiskPolicy(),
                base_policy_spec="anchor-risk-test",
                replay_paths=[path],
                margin_threshold=0.002,
                max_samples=1,
                max_per_stratum=1,
                first_per="seed-stratum",
            )

        self.assertEqual(len(corpus["samples"]), 1)
        self.assertEqual(corpus["samples"][0]["source_replay"], str(path))
        self.assertEqual(corpus["samples"][0]["base_action"], "right")
        self.assertEqual(corpus["samples"][0]["comparison_action"], "up")
        self.assertEqual(corpus["scan_stats"]["source"], "replay")

    def test_anchor_risk_replay_scan_can_start_after_prior_replays(self):
        state = anchor_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        replay = {
            "policy": "anchor-risk-test",
            "seed": 1,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": "right"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            first.write_text(json.dumps(replay))
            second.write_text(json.dumps({**replay, "seed": 2}))

            corpus = collect_anchor_risk_states_from_replays(
                base_policy=AnchorRiskPolicy(),
                base_policy_spec="anchor-risk-test",
                replay_paths=[first, second],
                margin_threshold=0.002,
                max_samples=1,
                max_per_stratum=1,
                first_per="none",
                max_replays=1,
                replay_start_index=1,
            )

        self.assertEqual(len(corpus["samples"]), 1)
        self.assertEqual(corpus["samples"][0]["source_replay"], str(second))
        self.assertEqual(corpus["scan_stats"]["available_replays"], 2)
        self.assertEqual(corpus["scan_stats"]["replays"], 1)
        self.assertEqual(corpus["scan_stats"]["replay_start_index"], 1)

    def test_anchor_risk_replay_scan_reuses_action_value_cache(self):
        state = anchor_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        replay = {
            "policy": "anchor-risk-test",
            "seed": 1,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": "right"},
                },
                {
                    "index": 2,
                    "state": state_payload(state, sim),
                    "move": {"action": "right"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            cache_path = Path(tmp) / "action_values.json"
            path.write_text(json.dumps(replay))

            cache: dict[str, list[dict[str, object]]] = {}
            policy = CountingAnchorRiskPolicy()
            corpus = collect_anchor_risk_states_from_replays(
                base_policy=policy,
                base_policy_spec="anchor-risk-test",
                replay_paths=[path],
                margin_threshold=0.002,
                max_samples=2,
                max_per_stratum=2,
                first_per="none",
                action_value_cache=cache,  # type: ignore[arg-type]
            )
            write_action_value_cache(cache_path, cache)  # type: ignore[arg-type]

            loaded = load_action_value_cache(cache_path)
            second_policy = CountingAnchorRiskPolicy()
            second = collect_anchor_risk_states_from_replays(
                base_policy=second_policy,
                base_policy_spec="anchor-risk-test",
                replay_paths=[path],
                margin_threshold=0.002,
                max_samples=2,
                max_per_stratum=2,
                first_per="none",
                action_value_cache=loaded,
            )

        self.assertEqual(len(corpus["samples"]), 1)
        self.assertEqual(corpus["scan_stats"]["action_value_cache_misses"], 1)
        self.assertEqual(corpus["scan_stats"]["action_value_cache_hits"], 1)
        self.assertGreater(policy.action_value_calls, 0)
        self.assertEqual(second_policy.action_value_calls, 0)
        self.assertEqual(second["scan_stats"]["action_value_cache_misses"], 0)
        self.assertEqual(second["scan_stats"]["action_value_cache_hits"], 2)

    def test_collect_geometry_risk_states_from_replay_frames(self):
        state = geometry_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        replay = {
            "policy": "geometry-risk-test",
            "seed": 1,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": "right"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(replay))

            corpus = collect_geometry_risk_states_from_replays(
                base_policy=GeometryRiskPolicy(),
                base_policy_spec="geometry-risk-test",
                replay_paths=[path],
                margin_threshold=0.01,
                max_samples=1,
                max_per_stratum=1,
                geometry_min_tile=3072,
                geometry_min_delta=10.0,
            )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["sample_mode"], "geometry-risk")
        self.assertEqual(sample["source_replay"], str(path))
        self.assertEqual(sample["base_action"], "right")
        self.assertEqual(sample["comparison_action"], "left")
        self.assertEqual(sample["top_two_actions"], ["right", "left"])
        self.assertGreater(sample["geometry"]["geometry_delta"], 10.0)
        self.assertEqual(corpus["scan_stats"]["sample_mode"], "geometry-risk")
        self.assertEqual(corpus["scan_stats"]["geometry_moves"], 1)
        self.assertEqual(corpus["scan_stats"]["geometry_disagreements"], 1)

    def test_collect_anchor_risk_states_from_state_records(self):
        state = anchor_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        records = [
            {
                "id": "anchor-record",
                "source_replay": "source/replay.json",
                "source_frame_index": 7,
                "seed": 1,
                "starter_tile": 1536,
                "state": state_payload(state, sim),
            }
        ]

        corpus = collect_anchor_risk_states_from_state_records(
            base_policy=AnchorRiskPolicy(),
            base_policy_spec="anchor-risk-test",
            state_records=records,
            margin_threshold=0.002,
            max_samples=1,
            max_per_stratum=1,
            first_per="replay-stratum",
        )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["source_replay"], "source/replay.json")
        self.assertEqual(sample["source_record_id"], "anchor-record")
        self.assertEqual(sample["source_frame_index"], 7)
        self.assertEqual(sample["base_action"], "right")
        self.assertEqual(sample["comparison_action"], "up")
        self.assertEqual(corpus["scan_stats"]["source"], "state_records")
        self.assertEqual(corpus["scan_stats"]["state_records"], 1)

    def test_collect_geometry_risk_states_from_state_records(self):
        state = geometry_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        records = [
            {
                "id": "geometry-record",
                "source_replay": "source/replay.json",
                "source_frame_index": 11,
                "seed": 1,
                "starter_tile": 1536,
                "state": state_payload(state, sim),
            }
        ]

        corpus = collect_geometry_risk_states_from_state_records(
            base_policy=GeometryRiskPolicy(),
            base_policy_spec="geometry-risk-test",
            state_records=records,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
            first_per="replay-stratum",
            geometry_min_tile=3072,
            geometry_min_delta=10.0,
        )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["source_record_id"], "geometry-record")
        self.assertEqual(sample["source_frame_index"], 11)
        self.assertEqual(sample["sample_mode"], "geometry-risk")
        self.assertEqual(sample["comparison_action"], "left")
        self.assertEqual(corpus["scan_stats"]["geometry_records"], 1)

    def test_support_chain_features_reward_half_max_support_near_max(self):
        state = support_chain_risk_state()
        down_board, _down_slots = simulate_base_move(state.board, DIRECTION_NAMES.index("down"))
        right_board, _right_slots = simulate_base_move(state.board, DIRECTION_NAMES.index("right"))

        down_features = support_chain_features(
            down_board,
            starter_tile=1536,
            support_min_tile=768,
            target_min_tile=3072,
        )
        right_features = support_chain_features(
            right_board,
            starter_tile=1536,
            support_min_tile=768,
            target_min_tile=3072,
        )

        self.assertTrue(down_features["target_support_adjacent_to_max"])
        self.assertGreater(down_features["score"], right_features["score"])

    def test_support_chain_features_raw_mode_keeps_starter_support(self):
        board = np.asarray(
            [
                [1536, 3072, 1536, 0],
                [768, 384, 192, 96],
                [48, 24, 12, 6],
                [3, 2, 1, 0],
            ],
            dtype=np.int32,
        )

        masked = support_chain_features(
            board,
            starter_tile=1536,
            support_min_tile=768,
            target_min_tile=3072,
        )
        raw = support_chain_features(
            board,
            starter_tile=1536,
            support_min_tile=768,
            target_min_tile=3072,
            mask_starter=False,
        )

        self.assertTrue(masked["mask_starter"])
        self.assertFalse(raw["mask_starter"])
        self.assertEqual(masked["count_target_support"], 1)
        self.assertFalse(masked["target_support_has_duplicate"])
        self.assertEqual(raw["count_target_support"], 2)
        self.assertTrue(raw["target_support_has_duplicate"])
        self.assertEqual(raw["highest_duplicate_tile"], 1536)

    def test_collect_support_chain_risk_states_from_replay_frames(self):
        state = support_chain_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        replay = {
            "policy": "support-chain-risk-test",
            "seed": 1,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": "right"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(replay))

            corpus = collect_support_chain_risk_states_from_replays(
                base_policy=SupportChainRiskPolicy(),
                base_policy_spec="support-chain-risk-test",
                replay_paths=[path],
                margin_threshold=0.01,
                max_samples=1,
                max_per_stratum=1,
                support_min_delta=100.0,
            )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["sample_mode"], "support-chain-risk")
        self.assertEqual(sample["source_replay"], str(path))
        self.assertEqual(sample["base_action"], "right")
        self.assertEqual(sample["comparison_action"], "down")
        self.assertEqual(sample["top_two_actions"], ["right", "down"])
        self.assertGreater(sample["support_chain"]["support_delta"], 100.0)
        self.assertEqual(corpus["scan_stats"]["sample_mode"], "support-chain-risk")
        self.assertEqual(corpus["scan_stats"]["support_moves"], 1)

    def test_collect_support_chain_risk_states_from_state_records(self):
        state = support_chain_risk_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        records = [
            {
                "id": "support-record",
                "source_replay": "source/replay.json",
                "source_frame_index": 13,
                "seed": 1,
                "starter_tile": 1536,
                "state": state_payload(state, sim),
            }
        ]

        corpus = collect_support_chain_risk_states_from_state_records(
            base_policy=SupportChainRiskPolicy(),
            base_policy_spec="support-chain-risk-test",
            state_records=records,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
            first_per="replay-stratum",
            support_min_delta=100.0,
        )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["source_record_id"], "support-record")
        self.assertEqual(sample["source_frame_index"], 13)
        self.assertEqual(sample["sample_mode"], "support-chain-risk")
        self.assertEqual(sample["comparison_action"], "down")
        self.assertEqual(corpus["scan_stats"]["support_records"], 1)

    def test_collect_top_two_states_from_state_records(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        records = [
            {
                "id": "top-two-record",
                "source_replay": "source/replay.json",
                "source_frame_index": 3,
                "seed": 1,
                "starter_tile": 1536,
                "target_tile": 3072,
                "outcome": "success",
                "moves_to_promotion": 4,
                "source_next_action": "left",
                "state": state_payload(state, sim),
            }
        ]

        corpus = collect_top_two_states_from_state_records(
            base_policy=FirstLegalNearTiePolicy(),
            base_policy_spec="top-two-test",
            state_records=records,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
            first_per="replay-stratum",
        )

        self.assertEqual(len(corpus["samples"]), 1)
        sample = corpus["samples"][0]
        self.assertEqual(sample["sample_mode"], "top-two")
        self.assertEqual(sample["source_record_id"], "top-two-record")
        self.assertEqual(sample["source_frame_index"], 3)
        self.assertEqual(sample["target_tile"], 3072)
        self.assertEqual(sample["outcome"], "success")
        self.assertEqual(sample["moves_to_promotion"], 4)
        self.assertEqual(sample["source_next_action"], "left")
        self.assertEqual(len(sample["top_two_actions"]), 2)
        self.assertEqual(sample["base_action"], sample["top_two_actions"][0])
        self.assertEqual(sample["comparison_action"], sample["top_two_actions"][1])
        self.assertEqual(corpus["scan_stats"]["sample_mode"], "top-two")
        self.assertEqual(corpus["scan_stats"]["accepted_samples"], 1)

    def test_collect_top_two_states_can_filter_low_value_ties(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(1), starter_tile=1536)
        records = [
            {
                "id": "top-two-record",
                "seed": 1,
                "starter_tile": 1536,
                "state": state_payload(state, sim),
            }
        ]

        corpus = collect_top_two_states_from_state_records(
            base_policy=FirstLegalNearTiePolicy(),
            base_policy_spec="top-two-test",
            state_records=records,
            margin_threshold=0.01,
            max_samples=1,
            max_per_stratum=1,
            min_top_value=101.0,
        )

        self.assertEqual(corpus["samples"], [])
        self.assertEqual(corpus["scan_stats"]["rejected"]["top_value_filter"], 1)

    def test_collect_from_replay_frames(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        action = int(sim.legal_actions(state)[0])
        replay = {
            "policy": "base",
            "seed": 7,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": DIRECTION_NAMES[action]},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(replay))

            corpus = collect_swing_states_from_replays(
                base_policy=FirstLegalNearTiePolicy(),
                base_policy_spec="base",
                comparison_policies=[("last", LastLegalPolicy())],
                replay_paths=[path],
                margin_threshold=0.01,
                max_samples=1,
                max_per_stratum=1,
            )

        self.assertEqual(len(corpus["samples"]), 1)
        self.assertEqual(corpus["scan_stats"]["source"], "replay")
        self.assertEqual(corpus["samples"][0]["base_action"], DIRECTION_NAMES[action])

    def test_collect_from_replay_can_use_policy_base_action(self):
        state = simple_state()
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        first_action = int(sim.legal_actions(state)[0])
        recorded_action = int(sim.legal_actions(state)[-1])
        replay = {
            "policy": "old_actor",
            "seed": 7,
            "starter_tile": 1536,
            "frames": [
                {"index": 0, "state": state_payload(state, sim), "move": None},
                {
                    "index": 1,
                    "state": state_payload(state, sim),
                    "move": {"action": DIRECTION_NAMES[recorded_action]},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(replay))

            corpus = collect_swing_states_from_replays(
                base_policy=FirstLegalNearTiePolicy(),
                base_policy_spec="base",
                comparison_policies=[("last", LastLegalPolicy())],
                replay_paths=[path],
                margin_threshold=0.01,
                max_samples=1,
                max_per_stratum=1,
                replay_base_action="policy",
            )

        self.assertEqual(len(corpus["samples"]), 1)
        self.assertEqual(corpus["samples"][0]["base_action"], DIRECTION_NAMES[first_action])
        self.assertEqual(corpus["samples"][0]["comparison_action"], DIRECTION_NAMES[recorded_action])

    def test_collect_from_replay_accepts_replay_scoped_first_per(self):
        state = simple_state()
        later_state = SimState(
            board=state.board.copy(),
            preview=state.preview,
            small_counts=state.small_counts.copy(),
            small_pos=state.small_pos,
            small_seen_total=state.small_seen_total,
            span_small_pos=state.span_small_pos,
            large_pending=state.large_pending,
            max_tile=state.max_tile,
            move_count=state.move_count + 1,
            game_over=state.game_over,
        )
        sim = ThreesSim(np.random.default_rng(7), starter_tile=1536)
        action = int(sim.legal_actions(state)[0])

        def replay_for(replay_state):
            return {
                "policy": "base",
                "seed": 7,
                "starter_tile": 1536,
                "frames": [
                    {"index": 0, "state": state_payload(replay_state, sim), "move": None},
                    {
                        "index": 1,
                        "state": state_payload(replay_state, sim),
                        "move": {"action": DIRECTION_NAMES[action]},
                    },
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            first.write_text(json.dumps(replay_for(state)))
            second.write_text(json.dumps(replay_for(later_state)))

            corpus = collect_swing_states_from_replays(
                base_policy=FirstLegalNearTiePolicy(),
                base_policy_spec="base",
                comparison_policies=[("last", LastLegalPolicy())],
                replay_paths=[first, second],
                margin_threshold=0.01,
                max_samples=2,
                max_per_stratum=2,
                first_per="replay-stratum",
            )

        self.assertEqual(len(corpus["samples"]), 2)
        self.assertEqual(corpus["scan_stats"]["accepted_scopes"], 2)

    def test_sample_scope_key_can_keep_first_per_stratum(self):
        features = {"phase": "late_1536", "stratum": "late_1536/high_corner_risk"}

        self.assertEqual(
            sample_scope_key("seed-policy-stratum", 10, "other", features),
            (10, "other", "late_1536/high_corner_risk"),
        )
        self.assertIsNone(sample_scope_key("none", 10, "other", features))

    def test_candidate_scope_key_can_keep_first_per_replay_stratum(self):
        features = {"phase": "late_1536", "stratum": "late_1536/high_corner_risk"}

        self.assertEqual(
            candidate_scope_key("replay-stratum", 10, "anchor_safe", features, "run/top/replay.json"),
            ("replay", "run/top/replay.json", "late_1536/high_corner_risk"),
        )

    def test_parse_filter_values_accepts_aliases(self):
        parsed = parse_filter_values(
            ["late,endgame"],
            allowed=("late_1536", "endgame_3072p"),
            aliases={"late": "late_1536", "endgame": "endgame_3072p"},
            label="phase",
        )

        self.assertEqual(parsed, {"late_1536", "endgame_3072p"})

    def test_load_samples_json_reads_prior_scan_payload(self):
        payload = {
            "base_policy": "base",
            "comparison_policies": ["other"],
            "seeds": "1:2",
            "margin_threshold": 0.002,
            "samples": [
                {
                    "id": "sample-1",
                    "features": {"stratum": "early/low"},
                    "top_two_actions": ["left", "right"],
                }
            ],
            "summary": {
                "scan_stats": {
                    "accepted_samples": 1,
                    "strata": {"early/low": 1},
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swing_labels.json"
            path.write_text(json.dumps(payload))

            samples, scan_stats, metadata = load_samples_json(path)

        self.assertEqual(samples[0]["id"], "sample-1")
        self.assertEqual(scan_stats["accepted_samples"], 1)
        self.assertEqual(metadata["source_comparison_policies"], ["other"])


if __name__ == "__main__":
    unittest.main()

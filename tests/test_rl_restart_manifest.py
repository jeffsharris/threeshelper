import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.record_replay import state_payload
from threes_rl.restart_manifest import build_restart_manifest
from threes_rl.ntuple import NtupleValue
from threes_rl.sim import SimState, ThreesSim, preview_from_label
from threes_rl.train_td import TDConfig, StartStateRecord, StartStateReservoir, load_start_state_records, play_episode


def state_for_stage(stage: int, move_count: int = 0) -> SimState:
    built = (192, 768, 1536, 3072)[stage]
    board = np.zeros((4, 4), dtype=np.int32)
    board[0, 0] = 1536
    board[0, 1] = built
    board[1, 0] = 1
    board[1, 1] = 2
    return SimState(
        board=board,
        preview=preview_from_label("gray"),
        small_counts={"red": 3, "blue": 2, "gray": 4},
        small_pos=3,
        small_seen_total=24,
        span_small_pos=3,
        large_pending=True,
        max_tile=1536 if stage < 3 else 3072,
        move_count=move_count,
        game_over=False,
    )


class RestartManifestTests(unittest.TestCase):
    def _replay(self, seed: int, *, include_endgame: bool = True):
        sim = ThreesSim(np.random.default_rng(seed), starter_tile=1536)
        initial = sim.reset()
        states = [initial, state_for_stage(0, 10), state_for_stage(1, 20), state_for_stage(2, 30)]
        if include_endgame:
            states.append(state_for_stage(3, 40))
        return {
            "policy": "corner2",
            "seed": seed,
            "starter_tile": 1536,
            "final_score": int(sum(int(value) for value in states[-1].board.reshape(-1))),
            "final_moves": states[-1].move_count,
            "final_max_tile": states[-1].max_tile,
            "frames": [
                {
                    "index": idx,
                    "state": state_payload(state, sim),
                    "move": None if idx == 0 else {"action": "left"},
                }
                for idx, state in enumerate(states)
            ],
        }

    def test_manifest_deduplicates_replay_copies_and_groups_root_ancestry(self):
        replay = self._replay(17)
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "copy.json"
            first.write_text(json.dumps(replay))
            second.write_text(json.dumps(replay))

            manifest = build_restart_manifest([first, second])

        self.assertEqual(manifest["counts"]["normal_start_replays"], 1)
        self.assertEqual(manifest["counts"]["duplicate_replay_copies"], 1)
        self.assertEqual(len({row["ancestry_key"] for row in manifest["records"]}), 1)

    def test_manifest_preserves_exact_state_and_cycle_fields(self):
        replay = self._replay(19)
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "replay.json"
            manifest_path = Path(tmp) / "manifest.json"
            replay_path.write_text(json.dumps(replay))
            manifest = build_restart_manifest([replay_path])
            manifest_path.write_text(json.dumps(manifest))

            loaded = load_start_state_records([str(manifest_path)], min_tile=0, starter_tile=1536)

        source = manifest["records"][2]
        matching = next(record for record in loaded if record.record_id == source["record_id"])
        np.testing.assert_array_equal(matching.state.board, np.asarray(source["state"]["board"], dtype=np.int32))
        self.assertEqual(matching.state.preview.label, source["state"]["preview"]["kind"])
        self.assertEqual(matching.state.small_counts, source["state"]["tile_cycle"]["small_counts"])
        self.assertEqual(matching.state.small_pos, source["state"]["tile_cycle"]["small_pos"])
        self.assertEqual(matching.state.span_small_pos, source["state"]["tile_cycle"]["span_small_pos"])
        self.assertEqual(matching.state.large_pending, source["state"]["tile_cycle"]["large_pending"])

    def test_manifest_keeps_success_and_failure_provenance_without_using_it_for_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            success_path = Path(tmp) / "success.json"
            failure_path = Path(tmp) / "failure.json"
            success_path.write_text(json.dumps(self._replay(21, include_endgame=True)))
            failure_path.write_text(json.dumps(self._replay(22, include_endgame=False)))
            manifest = build_restart_manifest([success_path, failure_path])

        outcomes = {row["trajectory_outcome"] for row in manifest["records"]}
        self.assertEqual(outcomes, {"success", "failure"})
        self.assertFalse(manifest["selection_uses_outcome"])


class AncestryBalancedSamplerTests(unittest.TestCase):
    def _records(self):
        records = []
        for stage in range(4):
            for ancestry, count in (("root-a", 9), ("root-b", 1)):
                for idx in range(count):
                    records.append(
                        StartStateRecord(
                            state=state_for_stage(stage, idx),
                            starter_tile=1536,
                            ancestry_key=f"{ancestry}-{stage}",
                            behavior_family="test-family",
                            trajectory_outcome="success" if ancestry == "root-a" else "failure",
                            record_id=f"{stage}-{ancestry}-{idx}",
                        )
                    )
        return records

    def test_sampler_balances_stage_then_ancestry_despite_frame_imbalance(self):
        reservoir = StartStateReservoir(self._records(), 1536, "ancestry_balanced")
        rng = np.random.default_rng(23)
        draws = [reservoir.sample_record(rng) for _ in range(40_000)]
        stage_counts = {stage: 0 for stage in range(4)}
        ancestry_counts = {(stage, ancestry): 0 for stage in range(4) for ancestry in ("root-a", "root-b")}
        for record in draws:
            self.assertIsNotNone(record)
            stage = int(record.record_id.split("-", 1)[0])
            ancestry = "root-a" if "root-a" in record.record_id else "root-b"
            stage_counts[stage] += 1
            ancestry_counts[(stage, ancestry)] += 1
        for count in stage_counts.values():
            self.assertLess(abs(count / 40_000 - 0.25), 0.015)
        for stage in range(4):
            share = ancestry_counts[(stage, "root-a")] / stage_counts[stage]
            self.assertLess(abs(share - 0.5), 0.025)

    def test_sampler_is_reproducible_and_reports_effective_ancestries(self):
        first = StartStateReservoir(self._records(), 1536, "ancestry_balanced")
        second = StartStateReservoir(self._records(), 1536, "ancestry_balanced")
        rng_a = np.random.default_rng(29)
        rng_b = np.random.default_rng(29)
        ids_a = [first.sample_record(rng_a).record_id for _ in range(200)]
        ids_b = [second.sample_record(rng_b).record_id for _ in range(200)]

        self.assertEqual(ids_a, ids_b)
        summary = first.sampling_summary()
        self.assertEqual(summary["restart_episodes"], 200)
        for stage in summary["stages"].values():
            self.assertEqual(stage["eligible_ancestries"], 2)
            self.assertGreater(stage["effective_ancestry_count"], 1.5)

    def test_exact_start_mix_alternates_normal_and_restart_episodes(self):
        reservoir = StartStateReservoir(self._records(), 1536, "ancestry_balanced")
        config = TDConfig(
            run_name="unused",
            games=4,
            pattern_set="tiny",
            start_state_prob=0.5,
            exact_start_mix=True,
            continuation_max_moves=1,
            max_moves=1,
        )
        model = NtupleValue.from_pattern_set("tiny")

        episodes = [play_episode(model, config, index, start_state_reservoir=reservoir) for index in range(1, 5)]

        self.assertEqual([episode.sampled_start is not None for episode in episodes], [False, True, False, True])


if __name__ == "__main__":
    unittest.main()

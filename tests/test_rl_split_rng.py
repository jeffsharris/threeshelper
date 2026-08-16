import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.eval import EvalStreamIds, run_game_with_optional_replay
from threes_rl.eval_stream_manifest import build_stream_manifest
from threes_rl.paired_eval_analysis import analyze
from threes_rl.r1b_freeze_d2 import freeze
from threes_rl.sim import ThreesSim


class FirstLegalPolicy:
    def __call__(self, state, sim, rng):
        return sim.legal_actions(state)[0]


class LastLegalPolicy:
    def __call__(self, state, sim, rng):
        return sim.legal_actions(state)[-1]


class RandomLegalPolicy:
    def __call__(self, state, sim, rng):
        legal = sim.legal_actions(state)
        return legal[int(rng.integers(len(legal)))]


class SplitRngTests(unittest.TestCase):
    def _run(self, policy, ids, max_moves=80):
        return run_game_with_optional_replay(
            policy,
            policy_name=policy.__class__.__name__,
            seed=7,
            starter_tile=1536,
            max_moves=max_moves,
            capture_replay=True,
            stream_ids=ids,
        )

    def test_identical_policy_and_streams_reproduce_exactly(self):
        ids = EvalStreamIds(deck_stream_id=101, slot_stream_id=202, policy_stream_id=303)
        first_result, first_replay = self._run(RandomLegalPolicy(), ids)
        second_result, second_replay = self._run(RandomLegalPolicy(), ids)

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_replay["frames"], second_replay["frames"])
        self.assertEqual(first_replay["rng_streams"]["evaluator_version"], "split_exogenous_v1")

    def test_changing_only_slot_stream_changes_positions_not_deck_values(self):
        first = ThreesSim.from_stream_ids(deck_stream_id=411, slot_stream_id=501, starter_tile=1536).reset()
        second = ThreesSim.from_stream_ids(deck_stream_id=411, slot_stream_id=502, starter_tile=1536).reset()

        self.assertEqual(sorted(first.board.reshape(-1).tolist()), sorted(second.board.reshape(-1).tolist()))
        self.assertEqual(first.preview, second.preview)
        self.assertFalse(np.array_equal(first.board, second.board))

    def test_changing_only_deck_stream_changes_tile_sequence(self):
        first = ThreesSim.from_stream_ids(deck_stream_id=601, slot_stream_id=701, starter_tile=1536).reset()
        second = ThreesSim.from_stream_ids(deck_stream_id=602, slot_stream_id=701, starter_tile=1536).reset()

        first_sequence = (sorted(first.board.reshape(-1).tolist()), first.preview.label)
        second_sequence = (sorted(second.board.reshape(-1).tolist()), second.preview.label)
        self.assertNotEqual(first_sequence, second_sequence)

    def test_divergent_trajectories_are_valid_and_reproducible(self):
        ids = EvalStreamIds(deck_stream_id=801, slot_stream_id=802, policy_stream_id=803)
        first_result, first_replay = self._run(FirstLegalPolicy(), ids)
        last_result, last_replay = self._run(LastLegalPolicy(), ids)
        first_again, first_replay_again = self._run(FirstLegalPolicy(), ids)
        last_again, last_replay_again = self._run(LastLegalPolicy(), ids)

        self.assertEqual(first_result, first_again)
        self.assertEqual(last_result, last_again)
        self.assertEqual(first_replay["frames"], first_replay_again["frames"])
        self.assertEqual(last_replay["frames"], last_replay_again["frames"])
        self.assertNotEqual(first_replay["frames"], last_replay["frames"])
        for replay in (first_replay, last_replay):
            for frame in replay["frames"][1:]:
                move = frame["move"]
                if move["inserted_pos"] is not None:
                    self.assertIn(move["inserted_pos"], move["eligible_positions"])

    def test_other_arm_policy_actions_do_not_change_repeated_arm(self):
        ids = EvalStreamIds(deck_stream_id=901, slot_stream_id=902, policy_stream_id=903)
        before, before_replay = self._run(FirstLegalPolicy(), ids)
        self._run(RandomLegalPolicy(), ids)
        after, after_replay = self._run(FirstLegalPolicy(), ids)

        self.assertEqual(before, after)
        self.assertEqual(before_replay["frames"], after_replay["frames"])

    def test_frozen_stream_manifest_has_disjoint_blocks(self):
        manifest = build_stream_manifest(namespace="test", block_sizes={"D0": 4, "D1": 5, "C": 6})
        all_ids = []
        for rows in manifest["blocks"].values():
            for row in rows:
                all_ids.extend([row["deck_stream_id"], row["slot_stream_id"], row["policy_stream_id"]])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(manifest["block_sizes"], {"D0": 4, "D1": 5, "C": 6})

    def test_r1b_d2_freeze_is_cross_manifest_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing_path = root / "existing.json"
            out_path = root / "d2.json"
            audit_path = root / "audit.json"
            existing = build_stream_manifest(
                namespace="existing",
                block_sizes={"D0": 4, "D1": 5, "C": 6},
            )
            existing_path.write_text(json.dumps(existing))

            audit = freeze(existing_path=existing_path, out_path=out_path, audit_path=audit_path)
            d2 = json.loads(out_path.read_text())

            self.assertEqual(audit["decision"], "PASS")
            self.assertEqual(audit["d2_games"], 512)
            self.assertFalse(any(audit["cross_block_collisions"].values()))
            self.assertFalse(audit["logical_seed_collisions"])
            self.assertEqual(len(d2["blocks"]["D2"]), 512)
            self.assertEqual(d2["blocks"]["D2"][0]["logical_seed"], 5_000_000)

    def test_paired_analysis_rejects_mismatched_streams(self):
        fields = [
            "block", "index", "logical_seed", "starter_tile", "deck_stream_id", "slot_stream_id",
            "policy_stream_id", "score", "score_minus_starter", "moves", "max_tile",
            "max_tile_excl_starter", "terminal_tile",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "baseline.csv"
            candidate = Path(tmp) / "candidate.csv"
            for path, deck in ((baseline, 1), (candidate, 2)):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "block": "D0", "index": 0, "logical_seed": 1, "starter_tile": 1536,
                            "deck_stream_id": deck, "slot_stream_id": 3, "policy_stream_id": 4,
                            "score": 60000, "score_minus_starter": 951, "moves": 20,
                            "max_tile": 1536, "max_tile_excl_starter": 192, "terminal_tile": False,
                        }
                    )
            with self.assertRaisesRegex(ValueError, "Mismatched paired stream"):
                analyze(baseline, candidate)


if __name__ == "__main__":
    unittest.main()

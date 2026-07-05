import tempfile
import unittest
from pathlib import Path

import numpy as np

from threes_rl.train_imitation import _split_sample_counts, load_dataset, save_dataset


class TrainImitationTests(unittest.TestCase):
    def test_split_sample_counts_covers_requested_samples(self):
        counts = _split_sample_counts(samples=23, workers=4, chunk_size=5)
        self.assertEqual(sum(counts), 23)
        self.assertEqual(counts, [5, 5, 5, 5, 3])

    def test_save_and_load_dataset_round_trips_arrays(self):
        obs = np.arange(30, dtype=np.float32).reshape(5, 6)
        actions = np.asarray([0, 1, 2, 3, 0], dtype=np.int64)
        masks = np.asarray(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
                [True, False, True, False],
            ],
            dtype=bool,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.npz"
            save_dataset(path, obs, actions, masks, expert="greedy", obs_encoder="board_only", starter_tile=None, seed=7)
            loaded_obs, loaded_actions, loaded_masks = load_dataset(path, samples=3)

        np.testing.assert_array_equal(loaded_obs, obs[:3])
        np.testing.assert_array_equal(loaded_actions, actions[:3])
        np.testing.assert_array_equal(loaded_masks, masks[:3])


if __name__ == "__main__":
    unittest.main()

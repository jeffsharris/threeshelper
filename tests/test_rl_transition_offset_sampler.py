import unittest

from threes_rl.transition_offset_sampler import parse_offsets, sample_offset_records


class TransitionOffsetSamplerTests(unittest.TestCase):
    def test_parse_offsets(self):
        self.assertEqual(parse_offsets("40,20,10"), {10, 20, 40})

    def test_samples_success_and_failure_offsets_with_root_cap(self):
        records = [
            {
                "id": "success-a",
                "target_tile": 3072,
                "outcome": "success",
                "moves_to_promotion": 10,
                "root_replay": "root-a",
                "root_seed": 1,
            },
            {
                "id": "success-a-duplicate",
                "target_tile": 3072,
                "outcome": "success",
                "moves_to_promotion": 10,
                "root_replay": "root-a",
                "root_seed": 1,
            },
            {
                "id": "success-a-20",
                "target_tile": 3072,
                "outcome": "success",
                "moves_to_promotion": 20,
                "root_replay": "root-a",
                "root_seed": 1,
            },
            {
                "id": "failure-b",
                "target_tile": 3072,
                "outcome": "failure",
                "moves_to_promotion": None,
                "moves_to_terminal": 10,
                "root_replay": "root-b",
                "root_seed": 2,
            },
            {
                "id": "wrong-target",
                "target_tile": 6144,
                "outcome": "failure",
                "moves_to_terminal": 10,
                "root_replay": "root-c",
                "root_seed": 3,
            },
        ]

        selected, rejected = sample_offset_records(
            records,
            offsets={10, 20, 40},
            target_filter={3072},
            max_per_root_offset=1,
        )

        self.assertEqual([record["id"] for record in selected], ["failure-b", "success-a", "success-a-20"])
        self.assertEqual([record["sample_offset"] for record in selected], [10, 10, 20])
        self.assertEqual(rejected["max_per_root_offset"], 1)
        self.assertEqual(rejected["target_filter"], 1)


if __name__ == "__main__":
    unittest.main()

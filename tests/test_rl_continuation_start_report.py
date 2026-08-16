import unittest

from threes_rl.continuation_start_report import (
    build_rows,
    feature_split,
    probability_band_summary,
    rank_bucket_summary,
    summarize,
)


class ContinuationStartReportTests(unittest.TestCase):
    def test_build_rows_summarizes_by_start_case(self):
        starts = [
            {
                "id": "a",
                "source_seed": 1,
                "source_frame_index": 10,
                "reachability_prob": 0.75,
                "reachability_rank": 3,
                "anchor_id": "anchor-a",
                "anchor_frame_index": 8,
                "frame_offset": 2,
                "synthetic_kind": "adjacent768",
                "corner_risk": "high_corner_risk",
                "empty_count": 2,
                "legal_count": 4,
                "raw_has_adjacent_768": True,
                "features": {"safe_smalls_until_large_possible": 7},
                "state": {
                    "board": [
                        [3072, 768, 768, 0],
                        [3, 6, 12, 24],
                        [48, 96, 192, 384],
                        [0, 0, 0, 0],
                    ]
                },
            },
            {
                "id": "b",
                "source_seed": 2,
                "source_frame_index": 20,
                "synthetic_kind": "adjacent768",
                "corner_risk": "high_corner_risk",
                "empty_count": 0,
                "legal_count": 3,
                "state": {
                    "board": [
                        [1, 3, 6, 12],
                        [24, 48, 96, 192],
                        [384, 768, 768, 1536],
                        [3072, 3, 6, 12],
                    ]
                },
            },
        ]
        continuations = [
            {"start_case_id": "a", "max_tile_excl_starter": 6144, "score_delta": 100},
            {"start_case_id": "a", "max_tile_excl_starter": 3072, "score_delta": 10},
            {"start_case_id": "b", "max_tile_excl_starter": 3072, "score_delta": 5},
        ]

        rows = build_rows(starts, continuations, target_tile=6144)
        summary = summarize(rows, target_tile=6144)
        split = feature_split(rows, threshold=0.5)

        self.assertEqual(rows[0]["start_case_id"], "a")
        self.assertEqual(rows[0]["hits"], 1)
        self.assertEqual(rows[0]["continuations"], 2)
        self.assertEqual(rows[0]["hit_rate"], 0.5)
        self.assertEqual(rows[0]["reachability_prob"], 0.75)
        self.assertEqual(rows[0]["reachability_rank"], 3.0)
        self.assertEqual(rows[0]["anchor_id"], "anchor-a")
        self.assertEqual(rows[0]["anchor_frame_index"], 8)
        self.assertEqual(rows[0]["frame_offset"], 2.0)
        self.assertEqual(rows[0]["raw_count_768"], 2)
        self.assertEqual(rows[0]["raw_has_adjacent_768"], 1.0)
        self.assertEqual(rows[0]["raw_count_3072"], 1)
        self.assertEqual(rows[1]["raw_count_1536"], 1)
        self.assertEqual(summary["continuations"], 3)
        self.assertEqual(summary["hits"], 1)
        self.assertEqual(summary["mixed_starts"], 1)
        self.assertEqual(summary["stable_failure_starts"], 1)
        self.assertEqual(summary["outcome_by_synthetic_kind"]["adjacent768"]["continuations"], 3)
        self.assertEqual(summary["outcome_by_synthetic_kind"]["adjacent768"]["hits"], 1)
        self.assertEqual(split["high_starts"], 1)
        self.assertEqual(split["low_starts"], 1)

        bands = probability_band_summary(rows, edges=[0.0, 0.5, 0.9, 1.000001])
        high_band = next(band for band in bands if band["probability_min"] == 0.5)
        low_band = next(band for band in bands if band["probability_min"] == 0.0)
        self.assertEqual(high_band["starts"], 1)
        self.assertEqual(high_band["hits"], 1)
        self.assertEqual(high_band["continuations"], 2)
        self.assertEqual(high_band["mixed_starts"], 1)
        self.assertEqual(low_band["starts"], 1)
        self.assertEqual(low_band["stable_failure_starts"], 1)

        rank_buckets = rank_bucket_summary(rows, bucket_size=1)
        self.assertEqual(rank_buckets[0]["rank_start"], 3)
        self.assertEqual(rank_buckets[0]["starts"], 1)
        self.assertEqual(rank_buckets[0]["hits"], 1)

    def test_uncontinued_starts_are_not_counted_as_failures(self):
        starts = [
            {
                "id": "a",
                "synthetic_kind": "real",
                "state": {"board": [[3072, 768, 0, 0], [3, 6, 12, 24], [48, 96, 192, 384], [0, 0, 0, 0]]},
            },
            {
                "id": "b",
                "synthetic_kind": "real",
                "state": {"board": [[3072, 768, 0, 0], [3, 6, 12, 24], [48, 96, 192, 384], [0, 0, 0, 0]]},
            },
        ]
        continuations = [
            {"start_case_id": "a", "max_tile_excl_starter": 3072, "score_delta": 10},
        ]

        rows = build_rows(starts, continuations, target_tile=6144)
        summary = summarize(rows, target_tile=6144)
        split = feature_split(rows, threshold=0.5)

        self.assertEqual(summary["starts"], 2)
        self.assertEqual(summary["stable_failure_starts"], 1)
        self.assertEqual(summary["outcome_by_synthetic_kind"]["real"]["starts"], 2)
        self.assertEqual(summary["outcome_by_synthetic_kind"]["real"]["stable_failure_starts"], 1)
        self.assertEqual(split["high_starts"], 0)
        self.assertEqual(split["low_starts"], 1)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from threes_rl.frontier_selector import rank_tuple, select_frontier_records, run_from_args


def record(
    record_id: str,
    *,
    root: str,
    game_over: bool = False,
    target_reached: bool = False,
    empty_count: int = 0,
    raw_count_768: int = 0,
    raw_count_1536: int = 0,
    raw_count_3072: int = 0,
    raw_dup: int = 0,
    raw_adj: int = 0,
    score_delta: int = 0,
) -> dict:
    return {
        "id": record_id,
        "kind": "rare_event_frontier_state",
        "target_reached": target_reached,
        "root_replay": root,
        "root_seed": int("".join(ch for ch in root if ch.isdigit()) or 0),
        "root_origin": "fresh",
        "root_policy_family": "fixture",
        "source_replay": f"{root}/source.json",
        "score_delta": score_delta,
        "features": {
            "empty_count": empty_count,
            "raw_count_768": raw_count_768,
            "raw_count_1536": raw_count_1536,
            "raw_count_3072": raw_count_3072,
            "raw_highest_duplicate_tile": raw_dup,
            "raw_highest_adjacent_pair_tile": raw_adj,
        },
        "state": {
            "game_over": game_over,
            "legal_actions": [] if game_over else ["up", "left"],
        },
    }


class FrontierSelectorTests(unittest.TestCase):
    def test_selects_live_diverse_records_for_next_target(self):
        records = [
            record("dead", root="r1", game_over=True, raw_count_1536=2, raw_dup=1536),
            record("weak", root="r1", raw_count_1536=1, raw_dup=96, empty_count=3),
            record("strong", root="r1", raw_count_1536=2, raw_dup=1536, empty_count=2, score_delta=10),
            record("other-root", root="r2", raw_count_1536=1, raw_dup=768, raw_adj=768, empty_count=4),
        ]

        selected, candidates, rejected = select_frontier_records(
            records,
            next_target="second_3072",
            max_records=4,
            max_per_root=1,
        )

        self.assertEqual([row["id"] for row in selected], ["strong", "other-root"])
        self.assertEqual(len(candidates), 3)
        self.assertEqual(rejected["not_live"], 1)
        self.assertEqual(rejected["max_per_root"], 1)
        self.assertEqual(selected[0]["frontier_selector"]["rank"], 1)

    def test_bottom_selection_can_filter_to_matched_roots(self):
        records = [
            record("top-r1", root="r1", raw_count_1536=2, raw_dup=1536, empty_count=3),
            record("bottom-r1", root="r1", raw_count_1536=1, raw_dup=12, empty_count=0),
            record("top-r2", root="r2", raw_count_1536=2, raw_dup=768, empty_count=2),
            record("other-root", root="r3", raw_count_1536=0, raw_dup=0, empty_count=4),
        ]

        selected, _candidates, rejected = select_frontier_records(
            records,
            next_target="raw_duplicate_1536",
            root_filter={"r1", "r2"},
            rank_order="bottom",
            max_records=2,
            max_per_root=1,
        )

        self.assertEqual([row["id"] for row in selected], ["bottom-r1", "top-r2"])
        self.assertEqual(rejected["root_filter"], 1)
        self.assertEqual(selected[0]["frontier_selector"]["rank_order"], "bottom")

    def test_air_survival_profile_prioritizes_air_before_support(self):
        airy = record("airy", root="r1", raw_count_1536=1, raw_dup=12, empty_count=4)
        clogged = record("clogged", root="r2", raw_count_1536=1, raw_dup=1536, empty_count=0)

        self.assertGreater(
            rank_tuple(clogged, next_target="raw_duplicate_1536", rank_profile="support"),
            rank_tuple(airy, next_target="raw_duplicate_1536", rank_profile="support"),
        )
        self.assertGreater(
            rank_tuple(airy, next_target="raw_duplicate_1536", rank_profile="air_survival"),
            rank_tuple(clogged, next_target="raw_duplicate_1536", rank_profile="air_survival"),
        )

    def test_raw_duplicate_768_ranking_profiles(self):
        airy = record("airy", root="r1", raw_count_768=1, raw_dup=384, empty_count=4)
        support = record("support", root="r2", raw_count_768=2, raw_dup=768, empty_count=0)

        self.assertGreater(
            rank_tuple(support, next_target="raw_duplicate_768", rank_profile="support"),
            rank_tuple(airy, next_target="raw_duplicate_768", rank_profile="support"),
        )
        self.assertGreater(
            rank_tuple(airy, next_target="raw_duplicate_768", rank_profile="air_survival"),
            rank_tuple(support, next_target="raw_duplicate_768", rank_profile="air_survival"),
        )

    def test_raw_adjacent_768_support_targets_rank_geometry(self):
        loose = record("loose", root="r1", raw_count_768=2, raw_dup=768, raw_adj=96, empty_count=4)
        adjacent = record("adjacent", root="r2", raw_count_768=2, raw_dup=768, raw_adj=768, empty_count=0)
        with_1536 = record(
            "with-1536",
            root="r3",
            raw_count_768=2,
            raw_count_1536=1,
            raw_dup=768,
            raw_adj=768,
            empty_count=0,
        )

        self.assertGreater(
            rank_tuple(adjacent, next_target="raw_adjacent_768", rank_profile="support"),
            rank_tuple(loose, next_target="raw_adjacent_768", rank_profile="support"),
        )
        self.assertGreater(
            rank_tuple(with_1536, next_target="raw_adjacent_768_with_1536", rank_profile="support"),
            rank_tuple(adjacent, next_target="raw_adjacent_768_with_1536", rank_profile="support"),
        )
        self.assertGreater(
            rank_tuple(loose, next_target="raw_adjacent_768_with_1536", rank_profile="air_survival"),
            rank_tuple(with_1536, next_target="raw_adjacent_768_with_1536", rank_profile="air_survival"),
        )

    def test_raw_one_1536_ranking_profiles(self):
        airy = record("airy", root="r1", raw_count_768=2, raw_count_1536=0, raw_dup=768, empty_count=4)
        support = record("support", root="r2", raw_count_768=1, raw_count_1536=1, raw_dup=1536, empty_count=0)

        self.assertGreater(
            rank_tuple(support, next_target="raw_one_1536", rank_profile="support"),
            rank_tuple(airy, next_target="raw_one_1536", rank_profile="support"),
        )
        self.assertGreater(
            rank_tuple(airy, next_target="raw_one_1536", rank_profile="air_survival"),
            rank_tuple(support, next_target="raw_one_1536", rank_profile="air_survival"),
        )

    def test_can_filter_by_target_reached_and_feature_bounds(self):
        records = [
            record("hit-air", root="r1", target_reached=True, raw_count_3072=2, raw_adj=3072, empty_count=3),
            record("hit-clog", root="r2", target_reached=True, raw_count_3072=2, raw_adj=3072, empty_count=0),
            record("miss-air", root="r3", target_reached=False, raw_count_3072=2, raw_adj=3072, empty_count=4),
            record("too-many-1536s", root="r4", target_reached=True, raw_count_1536=2, raw_count_3072=2, raw_adj=3072, empty_count=3),
        ]

        selected, _candidates, rejected = select_frontier_records(
            records,
            next_target="reached_6144",
            target_reached_filter="true",
            min_features={"empty_count": 1},
            max_features={"raw_count_1536": 1},
            max_records=8,
            max_per_root=1,
        )

        self.assertEqual([row["id"] for row in selected], ["hit-air"])
        self.assertEqual(rejected["target_reached_filter"], 1)
        self.assertEqual(rejected["min_feature_filter"], 1)
        self.assertEqual(rejected["max_feature_filter"], 1)

    def test_run_from_args_writes_selected_records(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "frontier_records.json"
            out_dir = tmp_path / "out"
            source.write_text(
                json.dumps(
                    [
                        record("a", root="r1", raw_count_3072=2, raw_adj=3072, empty_count=2),
                        record("b", root="r2", raw_count_3072=1, raw_adj=0, empty_count=4),
                    ]
                )
            )

            payload = run_from_args(
                Namespace(
                    state_json=[[source]],
                    next_target="reached_6144",
                    target_reached="all",
                    include_terminal=False,
                    min_feature=None,
                    max_feature=None,
                    root_filter_json=None,
                    rank_order="top",
                    rank_profile="air_survival",
                    max_records=4,
                    max_per_root=1,
                    max_per_source=0,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(payload["summary"]["selected_records"], 2)
            self.assertEqual(payload["summary"]["rank_profile"], "air_survival")
            self.assertEqual(payload["selected_records"][0]["id"], "b")
            self.assertTrue((out_dir / "frontier_selection.json").exists())
            self.assertTrue((out_dir / "records.json").exists())
            self.assertTrue((out_dir / "frontier_selection.html").exists())


if __name__ == "__main__":
    unittest.main()

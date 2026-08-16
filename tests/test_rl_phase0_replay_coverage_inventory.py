import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from threes_rl.phase0_replay_coverage_inventory import (
    failure_extractable_h40,
    inventory_replays,
    replay_action_signature,
    success_extractable_h40,
)


def frame(index: int, board: list[list[int]], *, action: str | None = "left", game_over: bool = False) -> dict:
    return {
        "index": index,
        "state": {
            "move_count": index,
            "board": board,
            "score": 59049 + index,
            "max_tile": max(max(row) for row in board),
            "game_over": game_over,
            "preview": {"kind": "blue", "label": "blue", "value": 1, "candidates": []},
            "tile_cycle": {
                "small_counts": {"1": 0, "2": 0, "3": 0},
                "small_pos": 8,
                "small_seen_total": 0,
                "span_small_pos": 0,
                "large_pending": False,
            },
        },
        "move": None if action is None else {"action": action},
    }


def initial_board() -> list[list[int]]:
    return [
        [1536, 1, 2, 3],
        [1, 2, 3, 1],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def replay(seed: int, policy: str, boards: list[list[list[int]]], *, final_score: int = 1000) -> dict:
    frames = [frame(index, board, action="left") for index, board in enumerate(boards)]
    frames[-1]["move"] = None
    frames[-1]["state"]["game_over"] = True
    return {
        "policy": policy,
        "seed": seed,
        "starter_tile": 1536,
        "final_score": final_score,
        "final_moves": len(frames) - 1,
        "final_max_tile": max(max(max(row) for row in board) for board in boards),
        "frames": frames,
    }


def success_replay(seed: int, policy: str) -> dict:
    boards = [initial_board()]
    boards.extend([[[1536, 768, 0, 0], [1, 2, 3, 1], [2, 0, 0, 0], [0, 0, 0, 0]]] * 3)
    boards.append([[1536, 1536, 0, 0], [1, 2, 3, 1], [2, 0, 0, 0], [0, 0, 0, 0]])
    return replay(seed, policy, boards, final_score=120000)


def failure_replay(seed: int, policy: str) -> dict:
    boards = [initial_board()]
    boards.extend([[[1536, 768, 0, 0], [1, 2, 3, 1], [2, 0, 0, 0], [0, 0, 0, 0]]] * 4)
    return replay(seed, policy, boards, final_score=90000)


class Phase0ReplayCoverageInventoryTests(unittest.TestCase):
    def test_extractability_detects_success_and_matched_failure(self):
        success_ok, success_info = success_extractable_h40(success_replay(1, "corner2"), target_tile=1536, horizon=40)
        failure_ok, failure_info = failure_extractable_h40(failure_replay(2, "corner2"), target_tile=1536, horizon=40)

        self.assertTrue(success_ok)
        self.assertEqual(success_info["reason"], "ok")
        self.assertTrue(failure_ok)
        self.assertEqual(failure_info["reason"], "ok")

    def test_inventory_groups_families_and_dedupes_replay_copies(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for name, payload in [
                ("corner_success", success_replay(1, "corner2")),
                ("corner_success_copy", success_replay(1, "corner2")),
                ("corner_failure", failure_replay(2, "corner2")),
                ("phase_success", success_replay(3, "ntuple_phaseblend_expectimax2")),
                ("phase_failure", failure_replay(4, "ntuple_phaseblend_expectimax2")),
            ]:
                path = root / name / "replay.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))
                paths.append(path)

            payload = inventory_replays(paths, min_roots=4, min_roots_per_outcome=2)

        self.assertEqual(payload["rejected"]["duplicate_replay_signature"], 1)
        self.assertEqual(payload["family_counts"]["corner2_lineage"]["extractable_h40_roots"], 2)
        self.assertEqual(payload["family_counts"]["phaseblend_incumbent_lineage"]["extractable_h40_roots"], 2)
        self.assertTrue(payload["corpus_ready_if_using_retained_replays"])
        self.assertTrue(payload["corpus_selectable_from_retained_replays"])

    def test_downsample_option_can_satisfy_family_share_without_new_roots(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            cases = [
                ("phase_success_1", success_replay(1, "ntuple_phaseblend_expectimax2")),
                ("phase_success_2", success_replay(2, "ntuple_phaseblend_expectimax2")),
                ("phase_failure_3", failure_replay(3, "ntuple_phaseblend_expectimax2")),
                ("phase_failure_4", failure_replay(4, "ntuple_phaseblend_expectimax2")),
                ("phase_failure_5", failure_replay(5, "ntuple_phaseblend_expectimax2")),
                ("corner_success_6", success_replay(6, "corner2")),
                ("corner_failure_7", failure_replay(7, "corner2")),
            ]
            for name, payload in cases:
                path = root / name / "replay.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload))
                paths.append(path)

            payload = inventory_replays(paths, min_roots=4, min_roots_per_outcome=1)

        self.assertFalse(payload["corpus_ready_if_using_retained_replays"])
        self.assertFalse(payload["readiness_checks"]["max_family_share"])
        self.assertTrue(payload["corpus_selectable_from_retained_replays"])
        self.assertEqual(payload["minimum_new_roots_needed_if_downsample_allowed"], 0)
        self.assertTrue(any(option["ready"] for option in payload["retained_downsample_options"]))

    def test_action_signature_changes_with_policy(self):
        first = replay_action_signature(success_replay(1, "corner2"))
        second = replay_action_signature(success_replay(1, "ntuple_phaseblend_expectimax2"))

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

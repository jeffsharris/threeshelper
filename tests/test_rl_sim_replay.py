import glob
import json
import unittest
from pathlib import Path

import state_hunt as sh
from threes_rl.sim import simulate_base_move, tokens_to_board


def session_is_consistent(events):
    for ev in events:
        if ev.get("type") != "observed_move":
            continue
        tc = ev.get("transition_check") or {}
        if tc.get("inserted_value") is None or not tc.get("valid"):
            continue
        if sh.label_for_insert_value(tc["inserted_value"]) != ev.get("before_preview_label"):
            return False
    return True


class SimReplayTests(unittest.TestCase):
    def test_valid_single_step_recorded_moves_replay(self):
        total = 0
        matched = 0
        failures = []
        for events_path in sorted(glob.glob("datasets/*/*/events.jsonl")):
            events = []
            with open(events_path) as fh:
                for line in fh:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not session_is_consistent(events):
                continue
            for ev in events:
                if ev.get("type") != "observed_move":
                    continue
                tc = ev.get("transition_check") or {}
                if (
                    not tc.get("valid")
                    or tc.get("inserted_value") is None
                    or tc.get("inserted_pos") is None
                    or int(ev.get("step_count", 1)) != 1
                    or ev.get("direction") is None
                    or ev.get("unknown_board")
                    or ev.get("unknown_preview")
                ):
                    continue
                before = sh.board_to_values(ev["before_board"])
                after = sh.board_to_values(ev["after_board"])
                if before is None or after is None:
                    continue
                total += 1
                shifted, eligible = simulate_base_move(before, ev["direction"])
                inserted_pos = tuple(tc["inserted_pos"])
                candidate = shifted.copy()
                if inserted_pos in eligible:
                    candidate[inserted_pos] = int(tc["inserted_value"])
                if inserted_pos in eligible and candidate.tolist() == after:
                    matched += 1
                else:
                    failures.append((events_path, ev.get("move_index"), ev.get("direction"), tc))
        if total == 0:
            self.skipTest("No replayable observed_move events found.")
        self.assertGreaterEqual(matched / total, 0.99, failures[:10])


if __name__ == "__main__":
    unittest.main()

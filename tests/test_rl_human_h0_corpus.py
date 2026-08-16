from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from threes_rl.human_h0_corpus import validate_replay
from threes_rl.human_play_server import HumanGameSession


class HumanH0CorpusTests(unittest.TestCase):
    def test_exact_human_session_replays_from_original_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = HumanGameSession(
                Path(directory),
                logical_seed=101,
                deck_stream_id=202,
                slot_stream_id=303,
            )
            for _ in range(5):
                session.move(session.sim.legal_actions(session.state)[0])
            replay = json.loads(session.replay_path.read_text())

            validation = validate_replay(replay)

            self.assertTrue(validation["exact"])
            self.assertEqual(validation["moves_replayed"], 5)


if __name__ == "__main__":
    unittest.main()

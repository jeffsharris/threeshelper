from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from threes_rl.r1b_c_preflight import tree_sha256


class R1bCPreflightTests(unittest.TestCase):
    def test_tree_hash_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.txt").write_text("two")
            (root / "a.txt").write_text("one")

            first = tree_sha256(root)
            second = tree_sha256(root)
            (root / "a.txt").write_text("changed")
            changed = tree_sha256(root)

            self.assertEqual(first, second)
            self.assertNotEqual(first["sha256"], changed["sha256"])
            self.assertEqual(first["files"], 2)


if __name__ == "__main__":
    unittest.main()

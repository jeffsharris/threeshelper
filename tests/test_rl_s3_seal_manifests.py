from __future__ import annotations

from threes_rl.s3_seal_manifests import compact_candidate, root_list_sha256


def test_root_list_hash_is_order_independent() -> None:
    assert root_list_sha256(["b", "a"]) == root_list_sha256(["a", "b"])


def test_compact_candidate_omits_state_only() -> None:
    record = {
        "record_id": "candidate",
        "root_cluster": "fresh:1:1536",
        "state_sha1": "abc",
        "state": {"board": [[0] * 4 for _ in range(4)]},
    }
    assert compact_candidate(record) == {
        "record_id": "candidate",
        "root_cluster": "fresh:1:1536",
        "state_sha1": "abc",
    }

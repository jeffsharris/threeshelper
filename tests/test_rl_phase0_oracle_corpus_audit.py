import unittest

from threes_rl.phase0_oracle_corpus_audit import (
    audit_records,
    behavior_policy_family,
    is_candidate_record,
    root_cap_candidates,
)


def record(
    idx: int,
    *,
    target_tile: int = 1536,
    outcome: str = "success",
    offset: int = 40,
    root_seed: int | None = None,
    policy: str = "ntuple_phaseblend_expectimax2",
) -> dict:
    seed = root_seed if root_seed is not None else idx
    return {
        "id": f"r{idx}",
        "target_tile": target_tile,
        "outcome": outcome,
        "moves_to_promotion": offset if outcome == "success" else None,
        "moves_to_terminal": offset if outcome == "failure" else None,
        "root_origin": "fresh",
        "root_seed": seed,
        "root_replay": f"replay_{seed}.json",
        "root_frame_index": 0,
        "root_policy_family": policy,
        "source_policy_family": policy,
        "source_replay": f"replay_{seed}.json",
        "source_next_action": "left",
        "state": {"board": [[0] * 4 for _ in range(4)]},
        "score_minus_starter": 1000 + idx,
    }


class Phase0OracleCorpusAuditTests(unittest.TestCase):
    def test_candidate_filter_requires_sentinel_target_horizon_and_state(self):
        accepted, reason = is_candidate_record(record(1), target_tile=1536, horizon=40)
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted")

        accepted, reason = is_candidate_record(record(2, target_tile=3072), target_tile=1536, horizon=40)
        self.assertFalse(accepted)
        self.assertEqual(reason, "target_tile")

        accepted, reason = is_candidate_record(record(3, offset=41), target_tile=1536, horizon=40)
        self.assertFalse(accepted)
        self.assertEqual(reason, "horizon")

        missing_state = record(4)
        del missing_state["state"]
        accepted, reason = is_candidate_record(missing_state, target_tile=1536, horizon=40)
        self.assertFalse(accepted)
        self.assertEqual(reason, "missing_state")

    def test_behavior_family_coalesces_current_incumbent_lineage(self):
        self.assertEqual(
            behavior_policy_family(record(1, policy="ntuple_phaseblend_expectimax2")),
            "phaseblend_incumbent_lineage",
        )
        self.assertEqual(
            behavior_policy_family(record(2, policy="current_incumbent_alias")),
            "phaseblend_incumbent_lineage",
        )
        self.assertEqual(
            behavior_policy_family(record(3, policy="ntuple_phaseblend1b_expectimax2")),
            "phaseblend_cheap_lineage",
        )
        self.assertEqual(behavior_policy_family(record(4, policy="corner2_depth2")), "corner2_lineage")

    def test_root_cap_selects_one_record_nearest_horizon_per_ancestry(self):
        same_root_far = record(1, offset=40, root_seed=7)
        same_root_near = record(2, offset=10, root_seed=7)
        other_root = record(3, offset=39, root_seed=8)

        selected = root_cap_candidates([same_root_near, same_root_far, other_root], horizon=40)

        self.assertEqual(len(selected), 2)
        by_seed = {row["root_seed"]: row for row in selected}
        self.assertEqual(by_seed[7]["id"], "r1")
        self.assertEqual(by_seed[8]["id"], "r3")

    def test_audit_reports_not_ready_when_one_behavior_family_dominates(self):
        rows = [record(i, outcome="success" if i < 12 else "failure") for i in range(20)]

        payload = audit_records(rows, source_paths=["records.json"])

        self.assertEqual(payload["root_capped_records"], 20)
        self.assertFalse(payload["corpus_ready_for_rollout_gate"])
        self.assertFalse(payload["diversity_checks"]["min_behavior_families"])
        self.assertFalse(payload["diversity_checks"]["max_family_share"])

    def test_audit_ready_with_two_balanced_families_and_both_outcomes(self):
        rows = []
        for i in range(10):
            rows.append(record(i, outcome="success", policy="ntuple_phaseblend_expectimax2"))
        for i in range(10, 20):
            rows.append(record(i, outcome="failure", policy="corner2_depth2"))

        payload = audit_records(rows, source_paths=["records.json"])

        self.assertTrue(payload["corpus_ready_for_rollout_gate"])
        self.assertEqual(payload["by_behavior_family"]["phaseblend_incumbent_lineage"], 10)
        self.assertEqual(payload["by_behavior_family"]["corner2_lineage"], 10)
        self.assertEqual(payload["by_outcome"], {"success": 10, "failure": 10})


if __name__ == "__main__":
    unittest.main()

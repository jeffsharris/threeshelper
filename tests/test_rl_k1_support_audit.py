from __future__ import annotations

import json
from pathlib import Path

import pytest

from threes_rl import k1_support_audit as audit


ROOT = Path(__file__).resolve().parents[1]


def _completion_rows() -> list[dict]:
    rows = []
    for family, _spec in audit.k1.FAMILY_SLATE:
        for index in range(36):
            rows.append({
                "behavior_family": family,
                "game_index": index,
                "complete": True,
            })
    return rows


def _retained_rows(counts: dict[str, list[int]]) -> list[dict]:
    return [
        {
            "behavior_family": family,
            "qualifying_state_count": count,
        }
        for family, values in counts.items()
        for count in values
    ]


def test_charter_and_spent_terminal_are_frozen() -> None:
    assert audit.sha256_path(audit.CHARTER_PATH) == (
        "a091900a3293d0274506bf7d25ff772ab80fa0a4d06c551d204c74890ebf2e27"
    )
    assert audit.sha256_path(audit.ROOT / "K1_TERMINAL_RESULT.json") == (
        audit.EXPECTED_K1_TERMINAL_SHA256
    )
    assert audit.sha256_path(audit.C2_TERMINAL) == (
        audit.EXPECTED_C2_TERMINAL_SHA256
    )
    assert not audit.OUTPUT_DIR.exists()


def test_qualifying_state_scan_reduces_values_to_boolean_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        {
            "root_ancestry": "root",
            "frame_index": 1,
            "state_sha256": "a" * 64,
            "empty_count": 2,
        },
        {
            "root_ancestry": "root",
            "frame_index": 2,
            "state_sha256": "b" * 64,
            "empty_count": 4,
        },
        {
            "root_ancestry": "root",
            "frame_index": 3,
            "state_sha256": "c" * 64,
            "empty_count": 4,
        },
    ]
    iterator = iter(candidates)
    monkeypatch.setattr(
        audit.k1,
        "_frame_candidate",
        lambda frame, root: next(iterator),
    )
    monkeypatch.setattr(
        audit.k1,
        "_incumbent_metadata",
        lambda row, incumbent: {
            "incumbent_margin": 123.0,
            "incumbent_legal_actions": ["up"],
            "trigger_reasons": {
                "low_empty": False,
                "low_margin": int(row["frame_index"]) == 2,
            },
        },
    )
    rows = audit.qualifying_states(
        {"frames": [{}, {}, {}]},
        root="root",
        incumbent=object(),
    )
    assert [row["frame_index"] for row in rows] == [1, 2]
    assert all(set(row) == {
        "root_ancestry", "frame_index", "state_sha256"
    } for row in rows)
    assert "margin" not in json.dumps(rows)
    assert "action" not in json.dumps(rows)
    assert "value" not in json.dumps(rows)


def test_deterministic_one_state_key_is_stable() -> None:
    row = {"frame_index": 12, "state_sha256": "d" * 64}
    first = audit._candidate_key("root", row)
    second = audit._candidate_key("root", dict(row))
    assert first == second
    changed = dict(row)
    changed["frame_index"] = 13
    assert audit._candidate_key("root", changed) != first


def test_family_support_reports_observed_and_unknown_bounds() -> None:
    families = [family for family, _spec in audit.k1.FAMILY_SLATE]
    summary = audit.summarize_family_support(
        completion_rows=_completion_rows(),
        retained_rows=_retained_rows({
            families[0]: [4, 5, 6, 7],
            families[1]: [4] * 11,
            families[2]: [4] * 9,
        }),
    )
    assert summary[families[0]]["observed_roots_by_threshold"] == {
        "ge_1": 4,
        "ge_2": 4,
        "ge_3": 4,
        "ge_4": 4,
    }
    assert summary[families[0]]["unobservable_roots"] == 32
    assert summary[families[0]][
        "ge1_maximum_possible_with_unknowns"
    ] == 36
    assert not summary[families[1]]["meets_minimum_12_observed"]


def test_decision_fails_closed_without_twelve_observed_roots() -> None:
    families = [family for family, _spec in audit.k1.FAMILY_SLATE]
    support = {
        family: {"observed_ge1_lower_bound": count}
        for family, count in zip(families, (4, 11, 9), strict=True)
    }
    decision, detail = audit.root_diverse_decision(
        support,
        signature_families=5,
        integrity_passes=True,
        alternative_trigger_support={
            "expectimax2": 0,
            "qd": 0,
        },
    )
    assert decision == "KILL_EXACT_DEPTH3_PROGRAM"
    assert not detail["checks"]["current_slate_minimum_12_each"]
    assert detail["maximum_family_share"] > 0.40


def test_decision_can_ready_balanced_observed_support() -> None:
    support = {
        family: {"observed_ge1_lower_bound": 12}
        for family, _spec in audit.k1.FAMILY_SLATE
    }
    decision, detail = audit.root_diverse_decision(
        support,
        signature_families=3,
        integrity_passes=True,
        alternative_trigger_support={},
    )
    assert decision == "READY_K2_ROOT_DIVERSE_PROPOSAL"
    assert detail["checks"]["current_slate_minimum_12_each"]
    assert detail["maximum_family_share"] == pytest.approx(1 / 3)


def test_distinct_alternatives_have_no_compatible_trigger_support() -> None:
    evidence = audit._alternative_evidence()
    assert evidence["genuine_signature_family_count"] >= 5
    assert evidence["qd_decision"] == "READY_QD_FAMILY_ADMISSION"
    assert evidence["k1_compatible_observed_trigger_roots"] == {
        "g1r_expectimax2": 0,
        "g1r_qd_static_archive_oneply_v2_terminal_schema": 0,
    }


def test_c2_and_k1_untouched_gates_remain_unopened() -> None:
    unopened = audit._unopened_audit()
    assert unopened["passes"]
    assert all(unopened["checks"].values())


def test_payload_hash_and_file_manifest_are_deterministic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.json"
    path.write_text("{}")
    first = audit.file_manifest(path)
    second = audit.file_manifest(path)
    assert first == second
    payload = audit.payload_with_hash({"value": 1})
    assert audit.verify_payload_hash(payload)
    payload["value"] = 2
    assert not audit.verify_payload_hash(payload)


def test_terminal_states_are_exactly_frozen() -> None:
    assert {
        "READY_K2_ROOT_DIVERSE_PROPOSAL",
        "KILL_EXACT_DEPTH3_PROGRAM",
    } == {
        "READY_K2_ROOT_DIVERSE_PROPOSAL",
        "KILL_EXACT_DEPTH3_PROGRAM",
    }

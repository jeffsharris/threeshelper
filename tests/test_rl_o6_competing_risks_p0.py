from __future__ import annotations

import ast
import copy
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from threes_rl import o6_competing_risks_p0 as p0
from threes_rl.o4_domain_safe_pair_option import (
    exhaustive_blocker_domain_proof,
)
from threes_rl.sim import ThreesSim


class ForbiddenSentinel(Mapping):
    def __init__(self, allowed: dict, forbidden: set[str]) -> None:
        self.allowed = allowed
        self.forbidden = forbidden

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"forbidden key accessed: {key}")
        return self.allowed[key]

    def __iter__(self):
        return iter(self.allowed)

    def __len__(self):
        return len(self.allowed)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"forbidden key accessed: {key}")
        return self.allowed.get(key, default)


def _synthetic_candidates(untouched_n: int) -> list[dict]:
    contract = p0.role_matrix_contract(untouched_n)
    rows: list[dict] = []
    counter = 0
    for family_index, family in enumerate(p0.FAMILY_ORDER):
        for target_index, target in enumerate(p0.TARGET_ORDER):
            required = sum(
                contract["matrices"][role][family_index][target_index]
                for role in p0.ROLE_ORDER
            )
            for _ in range(required):
                rows.append(
                    {
                        "ancestry": f"ancestry-{counter:05d}",
                        "family": family,
                        "state_hash": f"state-{counter:05d}",
                        "frame_index": counter,
                        "target": target,
                        "pair_coords": ((0, 0), (3, 3)),
                    }
                )
                counter += 1
    return rows


def _whitelist_payload() -> Mapping:
    cycle = ForbiddenSentinel(
        {
            "small_counts": {"red": 2, "blue": 3, "gray": 4},
            "small_pos": 3,
            "small_seen_total": 55,
            "span_small_pos": 7,
            "large_pending": True,
        },
        {"score", "final_score", "max_tile", "outcome"},
    )
    return ForbiddenSentinel(
        {
            "board": [
                [1536, 3, 6, 12],
                [48, 24, 48, 6144],
                [0, 3072, 2, 768],
                [0, 96, 192, 384],
            ],
            "preview": {"kind": "bonus", "candidates": [24, 48, 96]},
            "tile_cycle": cycle,
            "move_count": 120,
            "game_over": False,
        },
        {
            "score",
            "final_score",
            "max_tile",
            "legal_actions",
            "move",
            "action",
            "outcome",
            "milestone",
        },
    )


def _risk_fixture(statuses: list[str]) -> dict:
    rows = []
    for index, status in enumerate(statuses, start=1):
        merge, failure, live = p0.RISK_SCHEMA["row_mapping"][status]
        rows.append(
            {
                "t": index,
                "time_fraction": index / 40.0,
                "safe_merge_event": merge,
                "competing_failure_event": failure,
                "live_after_transition": live,
            }
        )
    absorbing = next(
        (status for status in statuses if status != "live"),
        None,
    )
    success_band = None
    if absorbing == "success":
        event_time = statuses.index("success") + 1
        success_band = next(
            index
            for index, (lower, upper) in enumerate(
                p0.RISK_SCHEMA["success_time_bands"]
            )
            if lower <= event_time <= upper
        )
    return {
        "schema_sha256": p0.RISK_SCHEMA_SHA256,
        "rows": rows,
        "absorbing_status": absorbing,
        "administrative_h40_censor": (
            absorbing is None and len(statuses) == 40
        ),
        "success_time_band": success_band,
    }


def test_charter_freezes_normalized_domain_and_exact_power_workload() -> None:
    text = (p0.REPO_ROOT / p0.CHARTER_PATH).read_text()
    assert "Exact board tile values" in text
    assert "are not subject to the `[0,1]` constraint" in text
    assert "batch exactly 16 datasets and 64 bootstraps" in text
    assert "No dataset, bootstrap, root, stratum, or ICC cell may be sampled" in text


def test_schema_hashes_are_literal_and_exact() -> None:
    assert p0.RISK_SCHEMA_SHA256 == (
        "b46e3cd785902bb5753c9066defe8fbcad3fc9bcef6666f2ad72833067042cac"
    )
    assert p0.SOURCE_STATE_SCHEMA_SHA256 == (
        "781d640f9f6eddf1a7ed75551f443651306308875b6fe265b7af19c7278b8671"
    )
    assert p0.PROTECTED_CONTRACT_SHA256 == (
        "4f6386e183bf7c7a3d2106dc6dfdbbc16044878b91241723962e9b65ea21d790"
    )
    assert p0.POWER_CONTRACT_SHA256 == (
        "a3966cbbae5981b3b180055833b562486eee5f93460ecfef6a44a067a73dfad4"
    )


def test_family_contract_is_distinct_and_fails_closed_on_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert p0.family_contract_audit()["passes"]
    changed = dict(p0.FAMILY_SIGNATURES)
    changed["o6_replaycal"] = changed["o6_corner2"]
    monkeypatch.setattr(p0, "FAMILY_SIGNATURES", changed)
    report = p0.family_contract_audit()
    assert not report["passes"]
    assert not report["checks"]["signatures_unique"]


@pytest.mark.parametrize("untouched_n", p0.POWER_ROOT_COUNTS)
def test_role_matrices_are_exact_and_below_family_cap(
    untouched_n: int,
) -> None:
    report = p0.role_matrix_contract(untouched_n)
    assert report["passes"]
    assert all(
        len(set(report["family_counts"][role])) == 1
        for role in p0.ROLE_ORDER
    )
    assert all(
        max(report["target_counts"][role])
        - min(report["target_counts"][role])
        <= 1
        for role in p0.ROLE_ORDER
    )


def test_allocator_is_deterministic_one_root_per_ancestry() -> None:
    candidates = _synthetic_candidates(192)
    forward = p0.allocate_roles_without_backtracking(
        candidates,
        untouched_n=192,
    )
    reverse = p0.allocate_roles_without_backtracking(
        list(reversed(candidates)),
        untouched_n=192,
    )
    assert forward["passes"]
    assert reverse["passes"]
    assert forward["selection_sha256"] == reverse["selection_sha256"]
    assert forward["selected_count"] == 384 + 96 + 192
    integrity = p0.partition_integrity(
        forward["selected"],
        expected_role_counts=p0.ROLE_COUNTS_BY_UNTOUCHED_N[192],
    )
    assert integrity["passes"]


def test_candidate_dedupe_uses_global_hash_argmin() -> None:
    rows = _synthetic_candidates(192)
    duplicate = dict(rows[0])
    duplicate["frame_index"] = 99_999
    duplicate["state_hash"] = "alternate"
    chosen = p0.dedupe_one_candidate_per_ancestry([rows[0], duplicate])
    expected = min(
        (rows[0], duplicate),
        key=p0.candidate_selection_key,
    )
    assert len(chosen) == 1
    assert chosen[0]["state_hash"] == expected["state_hash"]


def test_partition_integrity_rejects_ancestry_role_overlap() -> None:
    rows = [
        {
            "ancestry": f"a-{index}",
            "role": p0.ROLE_ORDER[index % 3],
            "family": p0.FAMILY_ORDER[index % 4],
        }
        for index in range(24)
    ]
    rows.append(
        {
            "ancestry": "a-0",
            "role": "development",
            "family": "o6_expectimax2",
        }
    )
    report = p0.partition_integrity(rows)
    assert not report["passes"]
    assert report["duplicate_ancestries"] == ["a-0"]
    assert not report["checks"]["ancestry_unique_across_roles"]


def test_partition_integrity_rejects_exactly_40_percent_family_share() -> None:
    families = (
        ["o6_corner2"] * 4
        + ["o6_expectimax2"] * 2
        + ["o6_parent_mc1000"] * 2
        + ["o6_replaycal"] * 2
    )
    rows = [
        {
            "ancestry": f"family-cap-{index}",
            "role": "train",
            "family": family,
        }
        for index, family in enumerate(families)
    ]
    report = p0.partition_integrity(rows)
    assert not report["passes"]
    assert not report["checks"]["family_cap_below_40_percent_each_role"]
    assert report["family_shares"]["train"]["o6_corner2"] == 0.40


def test_board_native_values_are_not_normalized_or_forbidden_fields_read() -> None:
    report = p0.current_state_round_trip(_whitelist_payload())
    assert report["stable"]
    assert report["payload"]["board"][0][0] == 1536
    assert report["payload"]["board"][1][3] == 6144
    assert p0.validate_normalized_values([0.0, 0.5, 1.0])["passes"]
    assert not p0.validate_normalized_values([0.0, 48.0])["passes"]


def test_current_state_exact_simulator_round_trip_uses_native_tile_domain() -> None:
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=26072907,
        slot_stream_id=26072908,
    )
    state = sim.reset()
    restored = sim.state_from_snapshot(
        state.board,
        state.preview,
        sim.tile_cycle_snapshot(state),
        state.move_count,
    )
    assert np.array_equal(restored.board, state.board)
    assert restored.preview == state.preview
    assert sim.tile_cycle_snapshot(restored) == sim.tile_cycle_snapshot(state)
    assert restored.move_count == state.move_count
    assert restored.game_over == state.game_over


def test_competing_risk_rows_round_trip_success_failure_and_censor() -> None:
    success = _risk_fixture(["live"] * 39 + ["success"])
    validated_success = p0.validate_competing_risk_fixture(success)
    assert validated_success["absorbing_status"] == "success"
    assert validated_success["success_time_band"] == 2
    assert not validated_success["administrative_h40_censor"]
    assert p0.competing_risk_round_trip(success)["stable"]

    failure = _risk_fixture(["live", "failure"])
    assert p0.validate_competing_risk_fixture(
        failure
    )["absorbing_status"] == "failure"
    assert p0.competing_risk_round_trip(failure)["stable"]

    censor = _risk_fixture(["live"] * 40)
    validated_censor = p0.validate_competing_risk_fixture(censor)
    assert validated_censor["absorbing_status"] is None
    assert validated_censor["administrative_h40_censor"]
    assert p0.competing_risk_round_trip(censor)["stable"]


def test_competing_risk_rows_reject_short_live_and_post_absorption() -> None:
    with pytest.raises(p0.O6ContractError, match="incomplete"):
        p0.validate_competing_risk_fixture(_risk_fixture(["live"] * 39))
    with pytest.raises(p0.O6ContractError, match="absorbing"):
        p0.validate_competing_risk_fixture(
            _risk_fixture(["success", "live"])
        )


def test_competing_risk_round_trip_rejects_non_one_hot_tamper() -> None:
    payload = _risk_fixture(["live", "success"])
    tampered = copy.deepcopy(payload)
    tampered["rows"][0]["live_after_transition"] = 0.5
    with pytest.raises(p0.O6ContractError, match="one-hot"):
        p0.competing_risk_round_trip(tampered)


def test_exhaustive_domain_proof_reproduces_all_coordinate_cases() -> None:
    proof = exhaustive_blocker_domain_proof()
    assert proof["passes"]
    assert proof["coordinate_pairs"] == 120
    assert proof["occupancy_cases"] == 43_296
    assert proof["minimum_density"] == 0.0
    assert proof["maximum_density"] == 1.0


def test_power_contract_is_estimate_only_and_exact() -> None:
    spec = p0.frozen_power_grid_spec()
    workload = p0.power_workload_estimate()
    assert spec["passes"]
    assert spec["row_count"] == 60
    assert workload["passes"]
    assert workload["cells"] == 60
    assert workload["datasets"] == 245_760
    assert workload["whole_root_bootstraps"] == 1_006_632_960
    assert workload["whole_root_index_draws"] == 338_228_674_560
    assert workload["dataset_batches_per_cell"] == 256
    assert workload["bootstrap_batches_per_dataset"] == 64
    assert not workload["power_executed"]


def test_power_math_primitives_are_deterministic_without_execution_loop() -> None:
    alpha, beta = p0.beta_parameters(p0.POWER_BASE_RATE, 0.15)
    assert alpha > 0.0
    assert beta > 0.0
    shifted = p0.odds_shift(np.asarray([p0.POWER_BASE_RATE]), 1.5)
    assert shifted.shape == (1,)
    assert shifted[0] > p0.POWER_BASE_RATE
    assert p0.power_seed(192, 1.50, 0.15) == p0.power_seed(
        192, 1.50, 0.15
    )
    control = np.asarray([[0] * 8, [0, 0, 0, 1, 0, 0, 0, 0]])
    treatment = np.asarray([[1] * 8, [0, 1, 0, 1, 0, 1, 0, 1]])
    estimate = p0.common_odds_ratio(
        control,
        treatment,
        np.asarray([0, 1]),
    )
    assert estimate > 1.0


def test_stream_windows_are_proposed_only_and_internally_disjoint() -> None:
    report = p0.stream_window_contract()
    assert report["passes"]
    assert not report["manifest_written"]
    assert not report["streams_reserved"]
    assert not report["streams_consumed"]
    assert not report["overlaps"]


def test_dependency_and_protected_audits_are_byte_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_json_parse(*args, **kwargs):
        raise AssertionError("protected JSON payload must not be parsed")

    monkeypatch.setattr(p0.json, "loads", forbidden_json_parse)
    dependencies = p0.dependency_audit()
    protected = p0.protected_source_audit()
    assert dependencies["passes"]
    assert protected["passes"]
    assert not dependencies["source_or_artifact_payloads_parsed"]
    assert not protected["content_inventory_executed"]
    assert not protected["json_payloads_parsed"]


def test_hash_contract_rejects_mutation_and_symlink(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real.json"
    real.write_text("one")
    expected = {real.name: p0.sha256_path(real)}
    assert p0._hash_contract(expected, repo_root=tmp_path)["passes"]
    real.write_text("two")
    assert not p0._hash_contract(expected, repo_root=tmp_path)["passes"]

    link = tmp_path / "link.json"
    link.symlink_to(real)
    expected_link = {link.name: p0.sha256_path(real)}
    assert not p0._hash_contract(expected_link, repo_root=tmp_path)["passes"]


def test_runner_has_no_scan_write_training_or_power_execution_path() -> None:
    source = (p0.REPO_ROOT / p0.RUNNER_PATH).read_text()
    tree = ast.parse(source)
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    forbidden_functions = {
        "scan",
        "prepare",
        "open",
        "execute",
        "reserve_streams",
        "generate_labels",
        "train",
        "fit",
        "simulate_power",
        "simulate_power_cell",
        "bootstrap_power",
    }
    assert function_names.isdisjoint(forbidden_functions)
    assert not any(
        name.startswith(("execute", "scan", "reserve", "generate", "train", "fit"))
        for name in function_names
    )
    assert "np.random" not in source
    assert "iter_eval_job_outputs" not in source
    assert "torch" not in source
    forbidden_calls = {
        "glob",
        "rglob",
        "walk",
        "write_text",
        "write_bytes",
        "mkdir",
        "rename",
        "replace",
        "touch",
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes.isdisjoint(forbidden_calls)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(module.startswith("threes_rl.") for module in imported_modules)
    assert "torch" not in imported_modules


@pytest.mark.parametrize(
    "verb",
    ("prepare", "open", "execute", "scan", "power", "reserve", "train"),
)
def test_cli_rejects_every_forbidden_verb(verb: str) -> None:
    with pytest.raises(SystemExit):
        p0.main([verb])


def test_cli_exposes_no_source_or_output_path_argument() -> None:
    help_text = p0._build_parser().format_help()
    assert "--out-dir" not in help_text
    assert "--source" not in help_text
    assert "--stream" not in help_text
    assert "--power" not in help_text
    assert "audit-preparation" in help_text


def test_full_preparation_audit_is_zero_work() -> None:
    report = p0.preparation_audit()
    assert report["passes"]
    assert report["decision"] == "CONTINUE_O6_SOURCE_PREPARATION_REVIEW"
    assert all(value == 0 for value in report["forbidden_work"].values())
    assert report["checks"]["source_scan_not_executed"]
    assert report["checks"]["root_selection_not_executed"]
    assert report["checks"]["historical_collision_scan_not_executed"]
    assert report["checks"]["power_simulation_not_executed"]
    assert report["zero_work"]["passes"]

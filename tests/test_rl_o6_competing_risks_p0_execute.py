from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from threes_rl import o6_competing_risks_p0 as prep
from threes_rl import o6_competing_risks_p0_execute as o6


def _passing_operational() -> dict:
    return {
        "nice": 10,
        "free_gib": 140.0,
        "output_bytes": 0,
        "process": {"passes": True},
        "services": {"passes": True},
        "checks": {"all": True},
        "passes": True,
    }


def _inventory() -> dict:
    rows = [
        {
            "path": "protected.json",
            "sha256": "a" * 64,
            "bytes": 2,
            "classification": "protected_governance_identity",
            "byte_stable": True,
        }
    ]
    return {
        "version": "fixture",
        "rows": rows,
        "row_count": 1,
        "classification_counts": {
            "protected_governance_identity": 1,
        },
        "inventory_sha256": o6.canonical_json_hash(rows),
        "checks": {"fixture": True},
        "passes": True,
        "payloads_parsed": False,
    }


def _source_payload() -> dict:
    return {
        "board": [
            [1536, 48, 0, 0],
            [0, 48, 0, 0],
            [3, 6, 12, 24],
            [1, 2, 3, 6],
        ],
        "preview": {"kind": "blue", "value": 1, "candidates": []},
        "tile_cycle": {
            "small_counts": {"red": 1, "blue": 2, "gray": 3},
            "small_pos": 7,
            "small_seen_total": 4,
            "span_small_pos": 1,
            "large_pending": False,
            "max_tile": 999999,
        },
        "move_count": 8,
        "game_over": False,
        "score": object(),
        "max_tile": object(),
        "legal_actions": object(),
        "move": object(),
        "action": object(),
        "outcome": object(),
    }


def _synthetic_candidates(untouched_n: int) -> list[dict]:
    rows = []
    counters: dict[tuple[str, int, str], int] = {}
    for quota in o6.allocation_cell_quotas(untouched_n):
        key = (
            str(quota["family"]),
            int(quota["target"]),
            str(quota["alignment"]),
        )
        start = counters.get(key, 0)
        for local in range(int(quota["required"])):
            index = start + local
            pair = (
                [[0, 0], [0, 1]]
                if quota["alignment"] == "aligned"
                else [[0, 0], [1, 1]]
            )
            rows.append(
                {
                    "ancestry": (
                        f"fresh:{quota['family']}:{quota['target']}:"
                        f"{quota['alignment']}:{index}"
                    ),
                    "family": quota["family"],
                    "target": quota["target"],
                    "alignment": quota["alignment"],
                    "state_hash": f"{index:064x}",
                    "frame_index": index,
                    "pair_coords": pair,
                }
            )
        counters[key] = start + int(quota["required"])
    return rows


def test_parent_hashes_are_literal_and_exact() -> None:
    report = o6.accepted_parent_audit()
    assert report["passes"]
    assert report["files"][str(prep.CHARTER_PATH)]["actual_sha256"] == (
        "2ee1e4273866f7f40376fb584e908f5a0e10e70446e2540f36bf320ac0edbb11"
    )


def test_output_names_and_namespace_are_separate() -> None:
    assert len(o6.OUTPUT_NAMES) == 16
    assert len(set(o6.OUTPUT_NAMES)) == 16
    assert "o6_competing_risks_p0_execution_v1" in str(o6.OUTPUT_DIR)
    assert o6.OUTPUT_DIR != prep.OUTPUT_DIR


def test_payload_hash_round_trip_and_tamper() -> None:
    payload = o6._payload_with_hash({"value": 3}, "self_sha")
    assert o6._verify_payload_hash(payload, "self_sha")
    payload["value"] = 4
    assert not o6._verify_payload_hash(payload, "self_sha")


def test_immutable_json_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    artifact = o6._write_immutable_json(path, {"x": 1})
    assert len(artifact["file_sha256"]) == 64
    with pytest.raises(FileExistsError):
        o6._write_immutable_json(path, {"x": 1})


def test_write_or_validate_resume_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "resume.json"
    first = o6._write_or_validate_json(
        path,
        {"x": 1},
        field="self_sha",
    )
    second = o6._write_or_validate_json(
        path,
        {"x": 1},
        field="self_sha",
    )
    assert not first["resumed"]
    assert second["resumed"]
    with pytest.raises(o6.O6IntegrityKill):
        o6._write_or_validate_json(
            path,
            {"x": 2},
            field="self_sha",
        )


def test_byte_inventory_classifies_without_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "runs/candidate").mkdir(parents=True)
    (tmp_path / "runs/governance").mkdir(parents=True)
    candidate = tmp_path / "runs/candidate/replay.json"
    governance = tmp_path / "runs/governance/root_manifest.json"
    candidate.write_text("{not parsed", encoding="ascii")
    governance.write_text("{also not parsed", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        o6,
        "CANDIDATE_PATTERNS",
        ("runs/**/replay.json",),
    )
    monkeypatch.setattr(
        o6,
        "PROTECTED_PATTERNS",
        ("runs/**/*.json",),
    )
    monkeypatch.setattr(o6, "LIVE_PATHS", frozenset())
    report = o6.build_byte_inventory(output_dir=tmp_path / "output")
    classes = {
        row["path"]: row["classification"] for row in report["rows"]
    }
    assert report["passes"]
    assert classes["runs/candidate/replay.json"] == "candidate_replay"
    assert classes["runs/governance/root_manifest.json"] == (
        "protected_governance_identity"
    )
    assert not report["payloads_parsed"]


def test_byte_inventory_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "runs").mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="ascii")
    (tmp_path / "runs/replay.json").symlink_to(target)
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        o6,
        "CANDIDATE_PATTERNS",
        ("runs/replay.json",),
    )
    monkeypatch.setattr(o6, "PROTECTED_PATTERNS", ())
    with pytest.raises(o6.O6DataHold, match="Symlink"):
        o6.build_byte_inventory(output_dir=tmp_path / "output")


def test_protected_path_precedes_candidate_glob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    human_dir = tmp_path / "threes_rl/runs/human_diagnostics"
    human_dir.mkdir(parents=True)
    replay = human_dir / "replay.json"
    replay.write_text("{must-not-be-parsed", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        o6,
        "CANDIDATE_PATTERNS",
        ("threes_rl/runs/**/replay.json",),
    )
    monkeypatch.setattr(
        o6,
        "PROTECTED_PATTERNS",
        ("threes_rl/runs/**/*.json",),
    )
    monkeypatch.setattr(o6, "LIVE_PATHS", frozenset())
    report = o6.build_byte_inventory(output_dir=tmp_path / "output")
    classification = report["rows"][0]["classification"]
    assert classification.startswith("protected_hash_only_")
    assert classification != "candidate_replay"


def test_inventory_validation_allows_live_rewrite_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable = tmp_path / "stable.json"
    live = tmp_path / "dashboard.json"
    stable.write_text("{}", encoding="ascii")
    live.write_text("{}", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        o6,
        "LIVE_PATHS",
        frozenset({"dashboard.json"}),
    )
    rows = [
        {
            "path": "stable.json",
            "sha256": o6.sha256_path(stable),
            "byte_stable": True,
        },
        {
            "path": "dashboard.json",
            "sha256": o6.sha256_path(live),
            "byte_stable": False,
        },
    ]
    inventory = {
        "rows": rows,
        "inventory_sha256": o6.canonical_json_hash(rows),
    }
    live.write_text('{"changed":true}', encoding="ascii")
    assert o6.validate_byte_inventory(inventory)["passes"]
    stable.write_text('{"changed":true}', encoding="ascii")
    assert not o6.validate_byte_inventory(inventory)["passes"]


def test_current_inventory_rejects_new_external_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    replay = runs / "replay.json"
    replay.write_text("{}", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(o6, "CANDIDATE_PATTERNS", ("runs/replay.json",))
    monkeypatch.setattr(o6, "PROTECTED_PATTERNS", ("runs/*.json",))
    monkeypatch.setattr(o6, "LIVE_PATHS", frozenset())
    frozen = o6.build_byte_inventory(output_dir=tmp_path / "output")
    (runs / "new_manifest.json").write_text("{}", encoding="ascii")
    report = o6.compare_inventory_to_current(
        frozen,
        output_dir=tmp_path / "output",
    )
    assert not report["passes"]
    assert report["new_paths"] == ["runs/new_manifest.json"]


def test_exclusion_union_reads_only_governance_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    governance = tmp_path / "root_manifest.json"
    forbidden = tmp_path / "episode.json"
    governance.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "root_cluster": "root-a",
                        "logical_seed": 123,
                    }
                ]
            }
        ),
        encoding="ascii",
    )
    forbidden.write_text("{not-json", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    inventory = {
        "rows": [
            {
                "path": governance.name,
                "sha256": o6.sha256_path(governance),
                "classification": "protected_governance_identity",
            },
            {
                "path": forbidden.name,
                "classification": "protected_hash_only_body",
            },
        ]
    }
    report = o6.build_exclusion_union(inventory)
    assert report["passes"]
    assert report["identities"]["root_cluster"] == ["root-a"]
    assert report["identities"]["logical_seed"] == [123]


def test_exclusion_union_rejects_unknown_identity_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "root_manifest.json"
    path.write_text(
        json.dumps({"mystery_stream_id": 123}),
        encoding="ascii",
    )
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    report = o6.build_exclusion_union(
        {
            "rows": [
                    {
                        "path": path.name,
                        "sha256": o6.sha256_path(path),
                        "classification": "protected_governance_identity",
                    }
            ]
        }
    )
    assert not report["passes"]
    assert report["unknown_identity_keys"] == ["mystery_stream_id"]


def test_exclusion_union_rejects_missing_bound_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "root_manifest.json"
    path.write_text("{}", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    with pytest.raises(o6.O6IntegrityKill):
        o6.build_exclusion_union(
            {
                "rows": [
                    {
                        "path": path.name,
                        "classification": "protected_governance_identity",
                    }
                ]
            }
        )


def test_candidate_scan_skips_protected_hash_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text("{must-not-be-parsed", encoding="ascii")
    replay_sha = o6.sha256_path(replay)
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    inventory = {
        "rows": [
            {
                "path": replay.name,
                "sha256": replay_sha,
                "classification": "candidate_replay",
            }
        ]
    }
    exclusion = {
        "identities": {
            "source_replay_sha256": [replay_sha],
            "replay_sha256": [],
        }
    }
    report = o6.scan_candidate_sources(inventory, exclusion)
    assert report["passes"]
    assert report["protected_hash_skip_count"] == 1
    assert report["source_count"] == 0
    assert not report["protected_replay_bodies_parsed"]


def test_candidate_scan_uses_frozen_source_copy_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("{}", encoding="ascii")
    right.write_text("{}", encoding="ascii")
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)

    def fake_candidates(
        _replay: dict,
        *,
        source_path: Path,
        source_sha256: str,
        exclusion: dict,
    ) -> tuple[list[dict], dict]:
        del exclusion
        row = {
            "ancestry": "fresh:1:1536",
            "family": "o6_corner2",
            "target": 48,
            "alignment": "aligned",
            "state_hash": source_sha256,
            "frame_index": 1,
            "pair_coords": [[0, 0], [0, 1]],
            "source_replay": source_path.name,
        }
        return [row], {
            "source_path": source_path.name,
            "source_sha256": source_sha256,
            "ancestry": "fresh:1:1536",
            "family": "o6_corner2",
            "policy_spec": "corner2",
        }

    monkeypatch.setattr(o6, "candidate_rows_from_replay", fake_candidates)
    inventory = {
        "rows": [
            {
                "path": path.name,
                "sha256": o6.sha256_path(path),
                "classification": "candidate_replay",
            }
            for path in (left, right)
        ]
    }
    exclusion = {
        "identities": {
            "source_replay_sha256": [],
            "replay_sha256": [],
        }
    }
    report = o6.scan_candidate_sources(inventory, exclusion)
    assert report["passes"]
    assert report["source_count"] == 2
    assert report["chosen_source_count"] == 1
    assert report["deduped_candidate_count"] == 1
    chosen = min(
        report["source_reports"],
        key=lambda row: row["source_copy_key"],
    )
    assert report["deduped_candidates"][0]["source_replay"] == (
        chosen["source_path"]
    )


def test_whitelisted_state_ignores_forbidden_fields() -> None:
    payload = _source_payload()
    state, identity = o6.whitelisted_sim_state(payload)
    assert state.max_tile == 1536
    assert identity["move_count"] == 8
    assert "score" not in identity
    assert "max_tile" not in identity
    assert np.max(state.board) > 1


def test_policy_spec_conflict_fails_closed() -> None:
    replay = {
        "policy": "corner2",
        "root_policy": "expectimax2",
    }
    with pytest.raises(o6.O6DataHold):
        o6._source_policy_spec(replay)


def test_family_policy_map_is_exact_and_unique() -> None:
    assert tuple(o6.FAMILY_POLICY_SPECS) == o6.FAMILY_ORDER
    assert len(set(o6.FAMILY_POLICY_SPECS.values())) == 4
    for family, spec in o6.FAMILY_POLICY_SPECS.items():
        assert o6._family_for_policy_spec(spec) == family


@pytest.mark.parametrize("n", prep.POWER_ROOT_COUNTS)
def test_alignment_quotas_match_role_contract(n: int) -> None:
    rows = o6.allocation_cell_quotas(n)
    assert len(rows) == 3 * 4 * 3 * 2
    assert sum(row["required"] for row in rows) == sum(
        prep.ROLE_COUNTS_BY_UNTOUCHED_N[n].values()
    )
    grouped: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (row["role"], row["family"], row["target"])
        grouped[key] = grouped.get(key, 0) + row["required"]
    parent = prep.role_matrix_contract(n)["matrices"]
    for role_index, role in enumerate(o6.ROLE_ORDER):
        for family_index, family in enumerate(o6.FAMILY_ORDER):
            for target_index, target in enumerate(o6.TARGET_ORDER):
                assert grouped[(role, family, target)] == (
                    parent[role][family_index][target_index]
                )


def test_exact_allocator_is_one_root_per_ancestry() -> None:
    rows = _synthetic_candidates(192)
    report = o6.allocate_candidate_design(rows, untouched_n=192)
    assert report["passes"]
    assert len(report["selected"]) == 672
    assert len({row["ancestry"] for row in report["selected"]}) == 672
    assert report["partition_integrity"]["passes"]


def test_allocator_fails_closed_on_one_missing_cell() -> None:
    rows = _synthetic_candidates(192)
    rows = [
        row
        for row in rows
        if not (
            row["family"] == "o6_corner2"
            and row["target"] == 48
            and row["alignment"] == "aligned"
        )
    ]
    report = o6.allocate_candidate_design(rows, untouched_n=192)
    assert not report["passes"]
    assert report["deficits"]


def test_stream_reservation_has_sixteen_fresh_ranges() -> None:
    report = o6.stream_reservation_contract()
    assert report["passes"]
    assert report["range_count"] == 16
    assert report["streams_consumed"] == 0
    starts = [row["start"] for row in report["ranges"]]
    assert starts == list(range(197_000_000_000, 213_000_000_000, 1_000_000_000))


def test_historical_collision_detects_value_inside_range() -> None:
    reservation = o6.stream_reservation_contract()
    clean = {
        "union_sha256": "u",
        "identities": {field: [] for field in o6.STREAM_FIELDS},
    }
    assert o6.historical_stream_collision_audit(
        clean,
        reservation,
    )["passes"]
    dirty = json.loads(json.dumps(clean))
    dirty["identities"]["logical_seed"] = [197_000_000_000]
    report = o6.historical_stream_collision_audit(dirty, reservation)
    assert not report["passes"]
    assert report["collisions"][0]["field"] == "logical_seed"


def test_stratified_bootstrap_preserves_each_stratum_count() -> None:
    strata = np.asarray([0, 0, 1, 1, 1, 2], dtype=np.int64)
    indices = o6.stratified_bootstrap_indices(
        np.random.default_rng(7),
        strata,
        20,
    )
    assert indices.shape == (20, 6)
    for row in indices:
        assert np.array_equal(strata[row], strata)


@pytest.mark.parametrize("n", prep.POWER_ROOT_COUNTS)
def test_power_strata_match_exact_untouched_allocation(n: int) -> None:
    strata = o6.power_strata_for_design(n)
    observed = {
        prep.POWER_STRATA[index]: count
        for index, count in Counter(strata.tolist()).items()
    }
    expected: dict[str, int] = {}
    for row in o6.allocation_cell_quotas(n):
        if row["role"] == "untouched_mechanism":
            label = f"T{row['target']}_{row['alignment']}"
            expected[label] = expected.get(label, 0) + row["required"]
    assert observed == expected
    assert len(strata) == n


def test_vectorized_common_or_matches_scalar_parent() -> None:
    rng = np.random.default_rng(4)
    control = rng.integers(0, 2, size=(12, 8), dtype=np.int8)
    treatment = rng.integers(0, 2, size=(12, 8), dtype=np.int8)
    strata = np.arange(12, dtype=np.int64) % 6
    indices = o6.stratified_bootstrap_indices(rng, strata, 9)
    vector = o6.vectorized_common_odds_ratio(
        control,
        treatment,
        strata,
        indices,
    )
    scalar = np.asarray(
        [
            prep.common_odds_ratio(
                control[row],
                treatment[row],
                strata,
            )
            for row in indices
        ]
    )
    assert np.allclose(vector, scalar, rtol=0.0, atol=1e-12)


def test_small_exact_power_batch_is_deterministic() -> None:
    kwargs = {
        "n": 192,
        "true_or": 1.5,
        "rho": 0.15,
        "dataset_start": 0,
        "dataset_count": 1,
        "bootstrap_count": 64,
        "bootstrap_batch_size": 64,
    }
    left = o6.simulate_power_dataset_batch(
        rng=np.random.default_rng(99),
        **kwargs,
    )
    right = o6.simulate_power_dataset_batch(
        rng=np.random.default_rng(99),
        **kwargs,
    )
    assert left == right
    assert len(left) == 1
    assert np.isfinite(left[0]["point_or"])
    assert np.isfinite(left[0]["lower95"])


def test_power_workload_is_full_not_approximated() -> None:
    report = o6.power_workload_contract()
    assert report["passes"]
    assert report["datasets"] == 245_760
    assert report["whole_root_bootstraps"] == 1_006_632_960
    assert report["whole_root_index_draws"] == 338_228_674_560


def test_power_database_initialization_and_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "power.sqlite3"
    o6.initialize_power_database(path, marker_sha256="m" * 64)
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM progress"
        ).fetchone()[0] == 60
        assert connection.execute(
            "SELECT COUNT(*) FROM results"
        ).fetchone()[0] == 0
    o6.initialize_power_database(path, marker_sha256="m" * 64)
    with pytest.raises(o6.O6IntegrityKill):
        o6.initialize_power_database(path, marker_sha256="x" * 64)


def test_power_database_integrity_rejects_progress_and_result_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "power.sqlite3"
    o6.initialize_power_database(path, marker_sha256="m" * 64)
    with sqlite3.connect(path) as connection:
        assert o6.audit_power_database(connection)["passes"]
        key = o6._power_cell_key(192, 1.25, 0.05)
        connection.execute(
            "UPDATE progress SET next_dataset=16 WHERE cell_key=?",
            (key,),
        )
        connection.commit()
        report = o6.audit_power_database(connection)
        assert not report["passes"]
        assert f"{key}:count" in report["result_failures"]

        connection.execute(
            "UPDATE progress SET next_dataset=0 WHERE cell_key=?",
            (key,),
        )
        connection.execute(
            "INSERT INTO results VALUES(?,?,?,?,?)",
            ("unknown", 0, 1.0, 1.0, 1),
        )
        connection.commit()
        report = o6.audit_power_database(connection)
        assert not report["passes"]
        assert report["unknown_result_cells"] == ["unknown"]


def test_power_database_integrity_rejects_nonfinite_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "power.sqlite3"
    o6.initialize_power_database(path, marker_sha256="m" * 64)
    key = o6._power_cell_key(192, 1.25, 0.05)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE progress SET next_dataset=16 WHERE cell_key=?",
            (key,),
        )
        connection.executemany(
            "INSERT INTO results VALUES(?,?,?,?,?)",
            [
                (
                    key,
                    index,
                    float("inf") if index == 0 else 1.0,
                    1.0,
                    1,
                )
                for index in range(16)
            ],
        )
        connection.commit()
        report = o6.audit_power_database(connection)
    assert not report["passes"]
    assert report["invalid_result_value_count"] == 1


def test_power_summary_is_incomplete_without_draws(
    tmp_path: Path,
) -> None:
    path = tmp_path / "power.sqlite3"
    o6.initialize_power_database(path, marker_sha256="m" * 64)
    with sqlite3.connect(path) as connection:
        report = o6.summarize_power_database(connection)
    assert not report["passes"]
    assert report["dataset_row_count"] == 0
    assert not report["selection"]["ready"]


def test_open_writes_only_marker_and_rejects_second_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        o6.TEST_EVIDENCE_NAME,
        o6.PREFLIGHT_LOCK_NAME,
        o6.PREFLIGHT_RESULT_NAME,
    ):
        (tmp_path / name).write_text("{}", encoding="ascii")
    monkeypatch.setattr(
        o6,
        "_validate_preflight_lock",
        lambda **_: (
            {
                "preflight_lock_payload_sha256": "l",
                "execute_command": "bound",
            },
            {"preflight_result_payload_sha256": "r"},
            {"test_evidence_payload_sha256": "e"},
        ),
    )
    monkeypatch.setattr(o6, "operational_audit", lambda **_: _passing_operational())
    monkeypatch.setattr(o6, "build_byte_inventory", lambda **_: _inventory())
    artifact = o6.open_execution(output_dir=tmp_path, jobs=1)
    assert Path(artifact["path"]).name == o6.OPENED_NAME
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [
            o6.TEST_EVIDENCE_NAME,
            o6.PREFLIGHT_LOCK_NAME,
            o6.PREFLIGHT_RESULT_NAME,
            o6.OPENED_NAME,
        ]
    )
    with pytest.raises(o6.O6IntegrityKill):
        o6.open_execution(output_dir=tmp_path, jobs=1)


def test_open_rejects_jobs_drift(tmp_path: Path) -> None:
    with pytest.raises(o6.O6IntegrityKill):
        o6.open_execution(output_dir=tmp_path, jobs=2)


def test_open_rejects_unknown_directory_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        o6.TEST_EVIDENCE_NAME,
        o6.PREFLIGHT_LOCK_NAME,
        o6.PREFLIGHT_RESULT_NAME,
    ):
        (tmp_path / name).write_text("{}", encoding="ascii")
    (tmp_path / "unexpected").mkdir()
    monkeypatch.setattr(
        o6,
        "_validate_preflight_lock",
        lambda **_: ({}, {}, {}),
    )
    with pytest.raises(o6.O6IntegrityKill, match="namespace differs"):
        o6.open_execution(output_dir=tmp_path)


def test_execute_operational_fault_seals_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / o6.OPENED_NAME
    marker_path.write_text("marker", encoding="ascii")
    marker = {"opened_payload_sha256": "m"}
    monkeypatch.setattr(o6, "_load_marker", lambda _: marker)
    monkeypatch.setattr(
        o6,
        "_execution_guard",
        lambda **_: (_ for _ in ()).throw(o6.O6DataHold("service")),
    )
    result = o6.execute(output_dir=tmp_path)
    payload = json.loads(Path(result["path"]).read_text(encoding="ascii"))
    assert payload["decision"] == "HOLD_O6_DATA_PREFLIGHT"


def test_execute_unknown_fault_seals_integrity_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / o6.OPENED_NAME
    marker_path.write_text("marker", encoding="ascii")
    marker = {"opened_payload_sha256": "m"}
    monkeypatch.setattr(o6, "_load_marker", lambda _: marker)
    monkeypatch.setattr(
        o6,
        "_execution_guard",
        lambda **_: (_ for _ in ()).throw(RuntimeError("corrupt")),
    )
    result = o6.execute(output_dir=tmp_path)
    payload = json.loads(Path(result["path"]).read_text(encoding="ascii"))
    assert payload["decision"] == "KILL_O6_P0_INTEGRITY"


def test_heavy_process_audit_ignores_ancestors_and_rejects_other_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(o6, "_ancestor_pids", lambda: {10, 20})

    def fake_run(args: tuple[str, ...], **_: object) -> SimpleNamespace:
        assert args[0] == "pgrep"
        return SimpleNamespace(stdout="10\n77\n")

    monkeypatch.setattr(o6.subprocess, "run", fake_run)
    report = o6.heavy_process_audit()
    assert not report["passes"]
    assert report["unrelated_candidate_pids"] == [77]
    assert 10 not in report["unrelated_candidate_pids"]


def test_recorder_health_accepts_status_ready_without_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "status": "ok",
            "active_sessions": 2,
            "advisor": {
                "status": "ready",
                "policy_file_sha256": o6.INCUMBENT_POLICY_SHA256,
            },
        }
    ).encode("utf-8")

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    monkeypatch.setattr(
        o6.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    report = o6.recorder_health()
    assert report["passes"]
    assert report["advisor_ready"]
    assert report["advisor_policy_exact"]
    assert "active_sessions" not in report
    assert not report["active_session_content_read"]


def test_service_audit_verifies_live_dashboard_top_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = tmp_path / "threes_rl/runs/dashboard/dashboard.json"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text(
        json.dumps(
            {
                "best_high_score": o6.TOP_THREE[0],
                "global_top_replays": [
                    {"score": score} for score in o6.TOP_THREE
                ],
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(o6, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(o6, "_socket_open", lambda _port: True)
    monkeypatch.setattr(
        o6,
        "recorder_health",
        lambda: {
            "passes": True,
            "active_session_content_read": False,
        },
    )
    report = o6.service_audit()
    assert report["passes"]
    assert report["dashboard"]["top_three"] == o6.TOP_THREE
    dashboard.write_text(
        json.dumps(
            {
                "best_high_score": 1,
                "global_top_replays": [{"score": 1}] * 3,
            }
        ),
        encoding="ascii",
    )
    assert not o6.service_audit()["passes"]


def test_total_power_runtime_sums_cells_and_open_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE progress("
        "active_seconds REAL NOT NULL,batch_opened_wall REAL)"
    )
    connection.executemany(
        "INSERT INTO progress VALUES(?,?)",
        [(3.0, None), (5.0, 90.0)],
    )
    monkeypatch.setattr(o6.time, "time", lambda: 100.0)
    assert o6.total_power_active_seconds(connection) == 18.0
    connection.close()


def test_runtime_journal_charges_resume_and_closes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = {"opened_payload_sha256": "m"}
    times = iter([100.0, 105.0, 112.0, 115.0])
    monkeypatch.setattr(o6.time, "time", lambda: next(times))
    opened = o6.begin_or_resume_runtime(
        output_dir=tmp_path,
        marker=marker,
        phase="scan",
    )
    assert opened["active_seconds"] == 0.0
    assert opened["resume_count"] == 0
    resumed = o6.begin_or_resume_runtime(
        output_dir=tmp_path,
        marker=marker,
        phase="scan",
    )
    assert resumed["active_seconds"] == 5.0
    assert resumed["resume_count"] == 1
    closed = o6.checkpoint_runtime(
        output_dir=tmp_path,
        marker=marker,
        next_phase=None,
    )
    assert closed["active_seconds"] == 12.0
    assert closed["phase"] is None
    assert closed["phase_opened_wall"] is None
    assert o6._load_hashed_json(
        tmp_path / o6.RUNTIME_NAME,
        field="runtime_payload_sha256",
    ) == closed


def test_runtime_journal_rejects_marker_drift(
    tmp_path: Path,
) -> None:
    marker = {"opened_payload_sha256": "m"}
    o6.begin_or_resume_runtime(
        output_dir=tmp_path,
        marker=marker,
        phase="scan",
    )
    with pytest.raises(o6.O6IntegrityKill):
        o6.checkpoint_runtime(
            output_dir=tmp_path,
            marker={"opened_payload_sha256": "other"},
            next_phase=None,
        )


def test_runtime_over_cap_is_operational_hold_not_integrity_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker_path = tmp_path / o6.OPENED_NAME
    marker_path.write_text("marker", encoding="ascii")
    marker = {"opened_payload_sha256": "m"}
    monkeypatch.setattr(o6, "_load_marker", lambda _: marker)
    monkeypatch.setattr(o6.time, "time", lambda: 100.0)
    o6.begin_or_resume_runtime(
        output_dir=tmp_path,
        marker=marker,
        phase="marker_validation",
    )
    runtime_path = tmp_path / o6.RUNTIME_NAME
    runtime = o6._load_hashed_json(
        runtime_path,
        field="runtime_payload_sha256",
    )
    runtime["active_seconds"] = o6.MAX_ACTIVE_HOURS * 3600.0 + 1.0
    runtime["phase_opened_wall"] = None
    runtime.pop("runtime_payload_sha256")
    o6._write_mutable_json(
        runtime_path,
        runtime,
        field="runtime_payload_sha256",
    )
    result = o6.execute(output_dir=tmp_path)
    payload = json.loads(Path(result["path"]).read_text(encoding="ascii"))
    assert payload["decision"] == "HOLD_O6_DATA_PREFLIGHT"


def test_zero_work_audit_never_calls_execution_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        o6,
        "build_byte_inventory",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("source inventory must remain closed")
        ),
    )
    monkeypatch.setattr(o6, "operational_audit", lambda **_: _passing_operational())
    report = o6.zero_work_preflight(output_dir=tmp_path / "absent")
    assert report["passes"]
    assert report["forbidden_work"]["corpus_source_scans"] == 0
    assert report["forbidden_work"]["power_datasets"] == 0


def test_cli_rejects_noncanonical_output(tmp_path: Path) -> None:
    with pytest.raises(o6.O6IntegrityKill):
        o6.dispatch(
            [
                "audit-zero-work",
                "--out-dir",
                str(tmp_path),
            ]
        )


def test_seal_preflight_requires_evidence(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        o6.seal_preflight(output_dir=tmp_path)

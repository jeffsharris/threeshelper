import copy
import json
from pathlib import Path

import pytest

from threes_rl import g3_scale_transfer_bootstrap_preflight as v1
from threes_rl import g3_scale_transfer_bootstrap_preflight_v2 as g3
from threes_rl.s3_power_preflight import sha256_path


def _tree(tmp_path: Path):
    root = tmp_path / "runs"
    source = root / "forensics" / "g2_fresh_transfer_acquisition_v1"
    self_dir = root / "forensics" / "g3_scale_transfer_bootstrap_preflight_v2"
    staging = root / "forensics" / "g3_scale_transfer_bootstrap_preflight_v2.staging"
    source.mkdir(parents=True)
    self_dir.mkdir(parents=True)
    staging.mkdir(parents=True)
    return root, source, self_dir, staging


def _classify(
    path: str,
    *,
    root: Path,
    source: Path,
    self_dirs: tuple[Path, ...],
    evidence=None,
):
    return g3.classify_match_path(
        path,
        search_root=root,
        input_namespace=source,
        self_namespaces=self_dirs,
        bound_evidence=evidence or {},
    )


def test_authoritative_v1_and_v2_amendment_hashes_reproduce():
    expected = {
        v1.CHARTER_PATH: v1.CHARTER_SHA256,
        v1.AMENDMENT_PATH: v1.AMENDMENT_SHA256,
        v1.IMPLEMENTATION_PATH: g3.V1_IMPLEMENTATION_SHA256,
        v1.TEST_PATH: g3.V1_TEST_SHA256,
        v1.TEST_EVIDENCE_PATH: g3.V1_TEST_EVIDENCE_SHA256,
        g3.V1_PREFLIGHT_PATH: g3.V1_PREFLIGHT_FILE_SHA256,
        g3.V1_RECORD_MANIFEST_PATH: g3.V1_RECORD_FILE_SHA256,
        g3.V1_STREAM_MANIFEST_PATH: g3.V1_STREAM_FILE_SHA256,
        g3.AMENDMENT_PATH: g3.AMENDMENT_SHA256,
    }
    assert {path: sha256_path(path) for path in expected} == expected


def test_v1_manifests_load_only_under_exact_hashes():
    preflight, records, streams, audit = g3.load_v1_manifests()
    assert audit["passes"]
    assert preflight["decision"] == "KILL_G3_PREFLIGHT_INTEGRITY"
    assert len(records["records"]) == g3.EXPECTED_RECORDS
    assert len(streams["rows"]) == g3.EXPECTED_TOTAL_PATHS
    assert streams["streams_consumed"] == 0


def test_internal_source_match_passes_and_is_hashed(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    path = source / "source.json"
    path.write_text("root-token")
    row = _classify(
        str(path),
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
    )
    assert row["category"] == "excluded_input"
    assert row["reasons"] == []
    assert row["sha256"] == sha256_path(path)


def test_exact_self_match_passes_but_forensics_sibling_fails(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    self_path = self_dir / "audit.json"
    sibling = root / "forensics" / "some_other_branch" / "copy.json"
    self_path.write_text("token")
    sibling.parent.mkdir()
    sibling.write_text("token")
    assert _classify(
        str(self_path),
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
    )["category"] == "excluded_self"
    row = _classify(
        str(sibling),
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
    )
    assert row["category"] == "external"
    assert "outside_exact_namespaces" in row["reasons"]


def test_outside_copy_and_prefix_lookalike_fail(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    outside = root / "external" / "copy.json"
    prefix = root / "forensics" / "g2_fresh_transfer_acquisition_v1-copy"
    outside.parent.mkdir()
    prefix.mkdir()
    outside.write_text("token")
    prefix_file = prefix / "copy.json"
    prefix_file.write_text("token")
    for path in (outside, prefix_file):
        row = _classify(
            str(path),
            root=root,
            source=source,
            self_dirs=(self_dir, staging),
        )
        assert row["category"] == "external"
        assert "outside_exact_namespaces" in row["reasons"]


def test_symlink_alias_fails_closed(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    target = source / "source.json"
    target.write_text("token")
    alias_dir = root / "external"
    alias_dir.mkdir()
    alias = alias_dir / "alias.json"
    try:
        alias.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    row = _classify(
        str(alias),
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
    )
    assert row["category"] == "external"
    assert "symlink_component" in row["reasons"]
    assert "resolved_path_differs" in row["reasons"]


def test_path_normalization_alias_fails_closed(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    target = source / "source.json"
    target.write_text("token")
    alias = str(root / "outside" / ".." / "forensics" /
                "g2_fresh_transfer_acquisition_v1" / "source.json")
    row = _classify(
        alias,
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
    )
    assert row["category"] == "external"
    assert "path_not_normalized_absolute" in row["reasons"]


def test_bound_v1_evidence_requires_exact_path_and_hash(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    evidence = root / "forensics" / "g3_v1" / "manifest.json"
    evidence.parent.mkdir()
    evidence.write_text("token")
    digest = sha256_path(evidence)
    good = _classify(
        str(evidence),
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
        evidence={evidence: digest},
    )
    bad = _classify(
        str(evidence),
        root=root,
        source=source,
        self_dirs=(self_dir, staging),
        evidence={evidence: "0" * 64},
    )
    assert good["category"] == "bound_v1_evidence"
    assert bad["category"] == "external"
    assert "bound_evidence_hash_mismatch" in bad["reasons"]


def test_classified_scan_reports_internal_and_external_separately(tmp_path):
    root, source, self_dir, staging = _tree(tmp_path)
    internal = source / "source.json"
    external = root / "external.json"
    internal.write_text("token")
    external.write_text("token")
    audit = g3._classified_scan(
        ["token"],
        search_root=root,
        input_namespace=source,
        self_namespaces=(self_dir, staging),
        bound_evidence={},
        match_paths=[str(internal), str(external)],
    )
    assert not audit["passes"]
    assert audit["category_counts"] == {
        "excluded_input": 1,
        "external": 1,
    }
    assert [row["path"] for row in audit["external_matches"]] == [
        str(external)
    ]


def test_panel_input_binding_covers_result_replay_and_state_files():
    result = g3._json(v1.TRANSFER_RESULT_PATH)
    sources = v1._transfer_sources(result)
    before = copy.deepcopy(sources)
    audit = g3.bind_transfer_panel_inputs(sources)
    assert audit["passes"]
    assert audit["unique_file_count"] == 65
    assert len(audit["rows"]) == 65
    assert {row["role"] for row in audit["rows"]} == {
        "sealed_transfer_result",
        "source_replay",
        "source_state",
    }
    assert sources == before


def test_panel_input_binding_rejects_outside_source(tmp_path):
    namespace = tmp_path / "source"
    namespace.mkdir()
    outside = tmp_path / "outside.json"
    state = namespace / "state.json"
    outside.write_text("{}")
    state.write_text("{}")
    sources = [
        {
            "root_cluster": "root",
            "source_replay": str(outside),
            "source_replay_sha256": sha256_path(outside),
            "source_state": str(state),
            "source_state_sha256": sha256_path(state),
        }
    ]
    audit = g3.bind_transfer_panel_inputs(
        sources, input_namespace=namespace
    )
    assert not audit["passes"]
    assert any(
        row["reason"] == "outside_exact_input_namespace"
        for row in audit["failures"]
    )


def test_staged_cost_decomposition_is_root_breadth_first_and_exact():
    _preflight, _records, streams, audit = g3.load_v1_manifests()
    assert audit["passes"]
    final = v1.cost_projection(g3.EXPECTED_TOTAL_PATHS)
    staged = g3.staged_cost_decomposition(
        streams["rows"], final_cost=final
    )
    assert staged["passes"]
    assert staged["stage_costs"]["E0"]["paths"] == 5_072
    assert staged["stage_costs"]["E1"]["paths"] == 15_216
    assert staged["stage_costs"]["E0"]["paths_by_partition"] == {
        "development": 944,
        "train": 3_902,
        "transfer_diagnostic": 226,
    }
    assert staged["stage_costs"]["E1"]["paths_by_partition"] == {
        "development": 2_832,
        "train": 11_706,
        "transfer_diagnostic": 678,
    }
    assert not staged["stage_costs"]["E0"]["authorized"]
    assert not staged["stage_costs"]["E1"]["authorized"]
    assert (
        staged["stage_costs"]["E0"]["projected_incremental_bytes"]
        + staged["stage_costs"]["E1"]["projected_incremental_bytes"]
        == final["projected_incremental_bytes"]
    )


def test_staged_cost_rejects_missing_replicate():
    _preflight, _records, streams, _audit = g3.load_v1_manifests()
    rows = streams["rows"][:-1]
    final = v1.cost_projection(len(rows))
    staged = g3.staged_cost_decomposition(rows, final_cost=final)
    assert not staged["passes"]
    assert not staged["checks"]["all_arms_have_exact_replicates_0_through_7"]


def test_test_evidence_audit_binds_current_files(tmp_path, monkeypatch):
    bound = tmp_path / "bound.py"
    bound.write_text("pass\n")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "version": "g3_v2_preflight_test_evidence_v1",
                "passes": True,
                "bound_files": [
                    {"path": str(bound), "sha256": sha256_path(bound)}
                ],
            }
        )
    )
    monkeypatch.setattr(g3, "TEST_EVIDENCE_PATH", evidence)
    assert g3._test_evidence_audit()["passes"]
    bound.write_text("changed\n")
    assert not g3._test_evidence_audit()["passes"]


def test_run_preflight_uses_fixed_staging_and_promotes_atomically(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    monkeypatch.setattr(g3, "OUTPUT_DIR", output)
    monkeypatch.setattr(g3, "STAGING_DIR", staging)
    payload = {
        "decision": "READY_G3_V2_BOOTSTRAP_LABELS",
        "canonical_payload_sha256": "placeholder",
    }
    auxiliary = {
        "passes": True,
        "canonical_payload_sha256": g3.canonical_sha256({"passes": True}),
    }
    monkeypatch.setattr(
        g3,
        "build_preflight_payload",
        lambda: (copy.deepcopy(payload), copy.deepcopy(auxiliary),
                 copy.deepcopy(auxiliary)),
    )
    result = g3.run_preflight()
    assert result["decision"] == "READY_G3_V2_BOOTSTRAP_LABELS"
    assert output.is_dir()
    assert not staging.exists()
    assert (output / "G3_V2_BOOTSTRAP_PREFLIGHT.json").is_file()
    with pytest.raises(FileExistsError):
        g3.run_preflight()


def test_run_preflight_failure_seals_fixed_staging(tmp_path, monkeypatch):
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    monkeypatch.setattr(g3, "OUTPUT_DIR", output)
    monkeypatch.setattr(g3, "STAGING_DIR", staging)

    def fail():
        raise RuntimeError("expected")

    monkeypatch.setattr(g3, "build_preflight_payload", fail)
    with pytest.raises(RuntimeError, match="expected"):
        g3.run_preflight()
    assert not output.exists()
    failure = json.loads((staging / "PREFLIGHT_FAILURE.json").read_text())
    assert failure["decision"] == "KILL_G3_V2_PREFLIGHT_INTEGRITY"
    assert failure["zero_forbidden_work"]["new_labels"] == 0


def test_output_and_staging_names_are_exact_and_not_broad():
    assert g3.OUTPUT_DIR.name == "g3_scale_transfer_bootstrap_preflight_v2"
    assert g3.STAGING_DIR.name == (
        "g3_scale_transfer_bootstrap_preflight_v2.staging"
    )
    assert g3.TRANSFER_INPUT_NAMESPACE.name == (
        "g2_fresh_transfer_acquisition_v1"
    )
    assert all(path.name != "forensics" for path in g3.SELF_NAMESPACES)

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from threes_rl import g3_e0_preflight as v1
from threes_rl import g3_e0_preflight_v2 as v2
from threes_rl.g3_e0_label_fit import payload_with_hash
from threes_rl.s3_power_preflight import sha256_path


def _ordinary_record() -> dict[str, object]:
    manifest = json.loads(v1.V1_RECORD_MANIFEST_PATH.read_text())
    return next(
        copy.deepcopy(record)
        for record in manifest["records"]
        if record["partition"] == "train"
    )


def test_v2_amendment_and_spent_v1_hashes_are_exact() -> None:
    assert sha256_path(v2.AMENDMENT_PATH) == v2.AMENDMENT_SHA256
    assert sha256_path(v2.FAILED_V1_LOCK_PATH) == (
        v2.FAILED_V1_LOCK_FILE_SHA256
    )
    lock = json.loads(v2.FAILED_V1_LOCK_PATH.read_text())
    assert lock["canonical_payload_sha256"] == (
        v2.FAILED_V1_LOCK_PAYLOAD_SHA256
    )
    assert lock["decision"] == "KILL_G3_E0_PREFLIGHT_INTEGRITY"
    assert lock["zero_forbidden_work"]["label_paths_generated"] == 0
    assert lock["zero_forbidden_work"]["scientific_models_fit"] == 0


def test_compact_record_adapter_restores_source_without_embedded_state() -> None:
    record = _ordinary_record()
    assert "state" not in record
    rows, audit = v2.validate_compact_ordinary_records([record])
    assert audit["passes"]
    assert len(rows) == 1
    assert rows[0]["state_sha1"] == record["state_sha1"]
    assert rows[0]["feature_rows_sha256"] == record["feature_rows_sha256"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_replay_sha256", "0" * 64, "hash mismatch"),
        ("root_cluster", "fresh:wrong:1536", "ancestry mismatch"),
        ("legal_action_ids", [0], "legal action IDs changed"),
        ("feature_rows_sha256", "0" * 64, "feature digest changed"),
    ],
)
def test_compact_adapter_fails_closed_on_bound_mismatch(
    field: str, value: object, message: str
) -> None:
    record = _ordinary_record()
    record[field] = value
    rows, audit = v2.validate_compact_ordinary_records([record])
    assert not rows
    assert not audit["passes"]
    assert message in audit["failures"][0]["error"]


def test_compact_adapter_rejects_embedded_state() -> None:
    record = _ordinary_record()
    record["state"] = {}
    rows, audit = v2.validate_compact_ordinary_records([record])
    assert not rows
    assert "unexpectedly embeds state" in audit["failures"][0]["error"]


def test_v2_input_audit_preserves_v1_and_upstream() -> None:
    audit = v2.v2_input_audit()
    assert audit["passes"]
    assert all(audit["checks"].values())


def test_v2_preflight_promotes_atomically_without_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "v2"
    simple = (
        {
            "version": v2.VERSION,
            "decision": "READY_G3_E0_LABEL_FIT_EXECUTION",
            "out_dir_resolved": str(out_dir.resolve()),
            "zero_forbidden_work": {
                "label_paths_generated": 0,
                "scientific_models_fit": 0,
            },
        },
        payload_with_hash({"records": []}),
        payload_with_hash({"rows": []}),
        payload_with_hash({"tasks": []}),
    )
    monkeypatch.setattr(v2, "build_preflight_payload", lambda out_dir: simple)
    lock = v2.run_preflight(out_dir)
    assert lock["decision"] == "READY_G3_E0_LABEL_FIT_EXECUTION"
    assert (out_dir / "preflight_lock.json").is_file()
    assert not out_dir.with_name(out_dir.name + ".staging").exists()
    assert not (out_dir / "ordinary_labels.sqlite3").exists()


def test_v2_preflight_failure_is_retained_in_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "v2"

    def fail(*, out_dir: Path) -> object:
        raise RuntimeError(f"adapter failed {out_dir.name}")

    monkeypatch.setattr(v2, "build_preflight_payload", fail)
    with pytest.raises(RuntimeError, match="adapter failed"):
        v2.run_preflight(out_dir)
    failure = json.loads(
        (
            out_dir.with_name(out_dir.name + ".staging")
            / "PREFLIGHT_FAILURE.json"
        ).read_text()
    )
    assert failure["decision"] == "KILL_G3_E0_PREFLIGHT_INTEGRITY"
    assert failure["zero_forbidden_work"]["label_paths_generated"] == 0


def test_v2_output_is_separate_and_fresh() -> None:
    assert v2.OUTPUT_DIR != v2.FAILED_V1_OUTPUT_DIR
    assert v2.OUTPUT_DIR.name == "g3_e0_label_fit_v2"
    assert v2.FAILED_V1_OUTPUT_DIR.is_dir()
    assert not v2.OUTPUT_DIR.exists()

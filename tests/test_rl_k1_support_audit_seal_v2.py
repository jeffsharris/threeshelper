import json
from pathlib import Path

import pytest

from threes_rl import k1_support_audit_seal_v2 as seal


def test_frozen_v1_serialization_defect_is_reproduced() -> None:
    source = json.loads(seal.SOURCE_PATH.read_text())
    embedded = source.pop("canonical_payload_sha256")
    assert seal.sha256_path(seal.SOURCE_PATH) == seal.EXPECTED_SOURCE_FILE_SHA256
    assert embedded == seal.EXPECTED_SOURCE_EMBEDDED_PAYLOAD_SHA256
    assert seal.canonical_json_hash(source) == seal.EXPECTED_SOURCE_BODY_SHA256
    assert embedded != seal.EXPECTED_SOURCE_BODY_SHA256


def test_reseal_preserves_every_scientific_field() -> None:
    source = json.loads(seal.SOURCE_PATH.read_text())
    source.pop("canonical_payload_sha256")
    payload = seal.build_reseal_payload()
    assert payload["scientific_payload"] == source
    assert payload["scientific_payload_sha256"] == seal.EXPECTED_SOURCE_BODY_SHA256
    assert payload["decision"] == source["decision"] == seal.EXPECTED_DECISION
    assert payload["scientific_fields_unchanged"]
    assert payload["source_replays_reopened"] == 0
    assert not payload["support_statistics_recomputed"]


def test_reseal_hash_survives_json_round_trip() -> None:
    payload = seal.build_reseal_payload()
    round_tripped = json.loads(json.dumps(payload, sort_keys=True))
    assert seal.verify_payload_hash(round_tripped)
    assert (
        seal.canonical_json_hash(round_tripped["scientific_payload"])
        == seal.EXPECTED_SOURCE_BODY_SHA256
    )


def test_reseal_records_zero_forbidden_work() -> None:
    payload = seal.build_reseal_payload()
    assert set(payload["forbidden_work"].values()) == {0}


def test_write_once_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "reseal.json"
    payload = seal.build_reseal_payload()
    seal.atomic_write_once(output, payload)
    assert seal.verify_payload_hash(json.loads(output.read_text()))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        seal.atomic_write_once(output, payload)


def test_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(seal.SOURCE_PATH.read_text() + "\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        seal.load_frozen_scientific_payload(changed)


def test_authoritative_output_is_separate_from_v1() -> None:
    assert seal.OUTPUT_PATH != seal.SOURCE_PATH
    assert "k1_support_audit_v2" in str(seal.OUTPUT_PATH)

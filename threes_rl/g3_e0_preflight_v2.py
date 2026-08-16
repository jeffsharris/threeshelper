"""Corrected compact-manifest preflight for the frozen G3 E0 screen."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import g3_e0_preflight as v1
from threes_rl import g3_scale_transfer_bootstrap_preflight as g3v1
from threes_rl import g3_scale_transfer_bootstrap_preflight_v2 as g3v2
from threes_rl.g2_scale_relational_hazard import schema_sha256
from threes_rl.g3_e0_label_fit import (
    CHARTER_PATH,
    CHARTER_SHA256,
    E0_REPLICATES,
    SCHEMA_SHA256,
    VERSION as RUNNER_VERSION,
    build_e0_tasks,
    canonical_sha256,
    feature_rows_for_record,
    json_object,
    payload_with_hash,
    task_coupling_audit,
    verify_payload_hash,
    write_immutable_json,
)
from threes_rl.replay_provenance import (
    GENUINE_ROOT_ORIGINS,
    replay_provenance,
)
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g3_e0_label_fit_preflight_v2"
AMENDMENT_PATH = Path(
    "threes_rl/G3_E0_LABEL_FIT_EXECUTION_CHARTER_AMENDMENT_V2_INTEGRITY.md"
)
AMENDMENT_SHA256 = (
    "1b0594d5c0cb55b7c5e11d24c64fa0f82d33952f6e4d4eab8eb5bdf429815fe7"
)
IMPLEMENTATION_PATH = Path("threes_rl/g3_e0_preflight_v2.py")
TEST_PATH = Path("tests/test_rl_g3_e0_preflight_v2.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/g3_e0_label_fit_v2_test_evidence.json"
)
OUTPUT_DIR = Path("threes_rl/runs/forensics/g3_e0_label_fit_v2")
FAILED_V1_OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/g3_e0_label_fit_v1"
)
FAILED_V1_LOCK_PATH = FAILED_V1_OUTPUT_DIR / "preflight_lock.json"
FAILED_V1_LOCK_FILE_SHA256 = (
    "73eee861d1ba964e4e9c2d24bffab29835f25fe2d435e7f9c665868a046dfa94"
)
FAILED_V1_LOCK_PAYLOAD_SHA256 = (
    "af4dcda1a2346e32450e7d37c6b989ae4b8ff139c625aca380d7fe424ceae45c"
)
V1_PREFLIGHT_IMPLEMENTATION_SHA256 = (
    "bb1c09b71060c1be1b6fb4bcfd03eb762950ea34d537b77f181f578c0b5cc627"
)
RUNNER_SHA256 = (
    "19d74a319459d75619f515fd9cdea03a126e1270046fb8e12ae367d43b2cc8b5"
)
V1_TEST_SHA256 = (
    "cd611547e86c6a65acc570440f4c94616549f2a419c439e3df669712a64687b1"
)
V1_TEST_EVIDENCE_SHA256 = (
    "0d4511eb4422f79304fa2dc17f54112e1f8cf3f0769fa086296c2ded59196ca9"
)
FUTURE_COMMAND = (
    "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
    ".venv/bin/python -m threes_rl.g3_e0_label_fit execute "
    "--out-dir threes_rl/runs/forensics/g3_e0_label_fit_v2 "
    "--preflight-lock "
    "threes_rl/runs/forensics/g3_e0_label_fit_v2/preflight_lock.json "
    "--jobs 1'"
)


def _physical_frame(
    replay: Mapping[str, Any], frame_index: int
) -> dict[str, Any]:
    matches = []
    for fallback, frame in enumerate(replay.get("frames", [])):
        if (
            isinstance(frame, dict)
            and int(frame.get("index", fallback)) == int(frame_index)
            and isinstance(frame.get("state"), dict)
        ):
            matches.append(frame["state"])
    if len(matches) != 1:
        raise ValueError(
            f"Expected one physical frame {frame_index}, found {len(matches)}"
        )
    return dict(matches[0])


def validate_compact_ordinary_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_cache: dict[str, dict[str, Any]] = {}
    replay_hashes: dict[str, str] = {}
    validated = []
    failures = []
    for record in records:
        record_id = str(record["record_id"])
        source_text = str(record["source_replay"])
        source_path = Path(source_text)
        try:
            if "state" in record:
                raise ValueError("Compact record unexpectedly embeds state")
            if source_text not in replay_cache:
                replay_hashes[source_text] = sha256_path(source_path)
                replay_cache[source_text] = json_object(source_path)
            if replay_hashes[source_text] != record["source_replay_sha256"]:
                raise ValueError("source replay hash mismatch")
            replay = replay_cache[source_text]
            provenance = replay_provenance(replay, source_path)
            if (
                provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS
                or provenance.get("root_origin") not in GENUINE_ROOT_ORIGINS
                or not provenance.get("replay_reset_invariant")
            ):
                raise ValueError("source replay is not a direct natural root")
            if canonical_ancestry_id(replay, source_path) != record["root_cluster"]:
                raise ValueError("canonical ancestry mismatch")
            starter = int(record["starter_tile"])
            payload = _physical_frame(
                replay, int(record["source_frame_index"])
            )
            if state_signature(payload, starter) != record["state_sha1"]:
                raise ValueError("source state signature mismatch")
            state = state_from_replay_payload(payload)
            validator = ThreesSim.from_stream_ids(
                deck_stream_id=2_026_072_551,
                slot_stream_id=2_026_072_552,
                starter_tile=starter,
            )
            legal_ids = validator.legal_actions(state)
            legal_names = [DIRECTION_NAMES[action] for action in legal_ids]
            if legal_ids != list(record["legal_action_ids"]):
                raise ValueError("legal action IDs changed")
            if legal_names != list(record["legal_actions"]):
                raise ValueError("legal action names changed")
            if not legal_ids:
                raise ValueError("selected source has no legal actions")
            _features, feature_sha = feature_rows_for_record(record)
            if feature_sha != record["feature_rows_sha256"]:
                raise ValueError("frozen feature digest changed")
            validated.append(
                {
                    "record_id": record_id,
                    "root_cluster": str(record["root_cluster"]),
                    "source_replay": source_text,
                    "source_replay_sha256": replay_hashes[source_text],
                    "source_frame_index": int(record["source_frame_index"]),
                    "state_sha1": str(record["state_sha1"]),
                    "legal_action_ids": legal_ids,
                    "legal_actions": legal_names,
                    "feature_rows_sha256": feature_sha,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "record_id": record_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    checks = {
        "all_records_validated": len(validated) == len(records),
        "no_failures": not failures,
        "one_validation_per_record":
            len({row["record_id"] for row in validated}) == len(records),
        "source_pointer_schema":
            all("state" not in record for record in records),
    }
    return validated, {
        "records": len(records),
        "validated_records": len(validated),
        "unique_sources": len(replay_cache),
        "failures": failures,
        "checks": checks,
        "passes": all(checks.values()),
    }


def validate_records_v2(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordinary = [
        dict(record)
        for record in records
        if record["partition"] != "transfer_diagnostic"
    ]
    transfer = [
        dict(record)
        for record in records
        if record["partition"] == "transfer_diagnostic"
    ]
    ordinary_validated, ordinary_audit = validate_compact_ordinary_records(
        ordinary
    )
    transfer_result = json_object(g3v1.TRANSFER_RESULT_PATH)
    transfer_sources = g3v1._transfer_sources(transfer_result)
    transfer_validated, transfer_audit = g3v1.validate_transfer_records(
        transfer_sources
    )
    ordinary_roots = {str(record["root_cluster"]) for record in ordinary}
    transfer_roots = {str(record["root_cluster"]) for record in transfer}
    checks = {
        "record_count_exact": len(records) == v1.EXPECTED_RECORDS,
        "ordinary_count_exact":
            len(ordinary) == v1.EXPECTED_ORDINARY_RECORDS,
        "transfer_count_exact":
            len(transfer) == v1.EXPECTED_TRANSFER_RECORDS,
        "ordinary_compact_sources_exact": ordinary_audit["passes"],
        "transfer_sources_exact": transfer_audit["passes"],
        "all_records_validated":
            len(ordinary_validated) + len(transfer_validated)
            == len(records),
        "expected_root_count":
            len(ordinary_roots | transfer_roots) == 384,
        "ordinary_transfer_overlap_zero":
            not ordinary_roots.intersection(transfer_roots),
    }
    return {
        "ordinary": ordinary_audit,
        "transfer": transfer_audit,
        "record_counts": dict(
            sorted(Counter(str(row["partition"]) for row in records).items())
        ),
        "root_count": len(ordinary_roots | transfer_roots),
        "checks": checks,
        "passes": all(checks.values()),
    }


def v2_input_audit() -> dict[str, Any]:
    inherited = v1.upstream_input_audit()
    locks = {
        AMENDMENT_PATH: AMENDMENT_SHA256,
        FAILED_V1_LOCK_PATH: FAILED_V1_LOCK_FILE_SHA256,
        Path("threes_rl/g3_e0_preflight.py"):
            V1_PREFLIGHT_IMPLEMENTATION_SHA256,
        Path("threes_rl/g3_e0_label_fit.py"): RUNNER_SHA256,
        Path("tests/test_rl_g3_e0_label_fit.py"): V1_TEST_SHA256,
        v1.TEST_EVIDENCE_PATH: V1_TEST_EVIDENCE_SHA256,
    }
    rows, files_exact = v1._locked_rows(locks)
    failed_lock = json_object(FAILED_V1_LOCK_PATH)
    checks = {
        "inherited_upstream_exact": inherited["passes"],
        "v1_failure_and_v2_amendment_files_exact": files_exact,
        "v1_failure_payload_exact":
            failed_lock.get("canonical_payload_sha256")
            == FAILED_V1_LOCK_PAYLOAD_SHA256
            and verify_payload_hash(failed_lock),
        "v1_decision_preserved":
            failed_lock.get("decision") == "KILL_G3_E0_PREFLIGHT_INTEGRITY",
        "v1_zero_work_preserved":
            failed_lock.get("zero_forbidden_work", {}).get(
                "label_paths_generated"
            )
            == 0
            and failed_lock.get("zero_forbidden_work", {}).get(
                "scientific_models_fit"
            )
            == 0,
    }
    return {
        "inherited": inherited,
        "additional_rows": rows,
        "checks": checks,
        "passes": all(checks.values()),
    }


def test_evidence_audit_v2() -> dict[str, Any]:
    original = v1.TEST_EVIDENCE_PATH
    try:
        v1.TEST_EVIDENCE_PATH = TEST_EVIDENCE_PATH
        return v1.test_evidence_audit()
    finally:
        v1.TEST_EVIDENCE_PATH = original


def build_preflight_payload(
    *,
    out_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if schema_sha256() != SCHEMA_SHA256:
        raise ValueError("G2 feature schema changed")
    inputs = v2_input_audit()
    spent_v1, record_manifest, stream_manifest, reuse = (
        g3v2.load_v1_manifests()
    )
    records = [dict(row) for row in record_manifest["records"]]
    streams = [dict(row) for row in stream_manifest["rows"]]
    tasks = build_e0_tasks(records, streams)
    record_audit = validate_records_v2(records)
    coupling = task_coupling_audit(tasks)
    collision = v1.historical_collision_audit(
        tasks,
        excluded_exact_directories=(
            g3v2.V1_OUTPUT_DIR,
            g3v2.OUTPUT_DIR,
            FAILED_V1_OUTPUT_DIR,
        ),
    )
    exact_exclusions = {
        str(g3v2.V1_OUTPUT_DIR.resolve()),
        str(g3v2.OUTPUT_DIR.resolve()),
        str(FAILED_V1_OUTPUT_DIR.resolve()),
    }
    by_partition = dict(
        sorted(Counter(str(task["partition"]) for task in tasks).items())
    )
    task_checks = {
        "task_count_exact": len(tasks) == v1.EXPECTED_E0_PATHS,
        "partition_counts_exact":
            by_partition == v1.EXPECTED_E0_BY_PARTITION,
        "replicates_exact":
            {int(task["replicate"]) for task in tasks}
            == set(E0_REPLICATES),
        "stream_coupling_exact": coupling["passes"],
        "stream_collisions_zero": collision["passes"],
        "collision_exclusions_exact":
            set(collision["excluded_exact_directories"])
            == exact_exclusions,
    }

    transfer_lock = json_object(g3v1.TRANSFER_PREFLIGHT_LOCK_PATH)
    incumbent = g3v1._verify_incumbent_artifacts(transfer_lock)
    v2_ready = json_object(g3v2.OUTPUT_DIR / "G3_V2_BOOTSTRAP_PREFLIGHT.json")
    stage_cost = v2_ready["staged_cost_decomposition"]["stage_costs"]["E0"]
    cost_checks = {
        "path_count_exact":
            int(stage_cost["paths"]) == v1.EXPECTED_E0_PATHS,
        "runtime_at_most_18h":
            float(stage_cost["projected_runtime_hours"])
            <= v1.MAX_PROJECTED_RUNTIME_HOURS,
        "storage_below_4gib":
            int(stage_cost["projected_incremental_bytes"])
            < v1.MAX_INCREMENTAL_BYTES,
    }
    disk = v1.disk_audit(v1.WORKSPACE_ROOT)
    operations = v1.process_and_service_audit()
    tests = test_evidence_audit_v2()

    integrity_checks = {
        "v1_failure_and_upstream_inputs_exact": inputs["passes"],
        "v1_manifests_exact": reuse["passes"],
        "compact_record_adapter_exact": record_audit["passes"],
        "task_contract_exact": all(task_checks.values()),
        "incumbent_payloads_exact": incumbent["passes"],
        "feature_schema_exact": schema_sha256() == SCHEMA_SHA256,
        "focused_and_regression_tests_pass": tests["passes"],
    }
    readiness_checks = {
        "cost_contract_passes": all(cost_checks.values()),
        "disk_above_100gib": disk["above_hard_minimum"],
        "disk_above_120gib_target": disk["above_target"],
        "services_process_and_nice_pass": operations["passes"],
        "output_dir_fresh": not out_dir.exists(),
        "no_existing_e0_work": not any(
            (out_dir / name).exists()
            for name in (
                "ordinary_labels.sqlite3",
                "transfer_labels.sqlite3",
                "checkpoint",
                "G3_E0_EXECUTION_OPENED.json",
            )
        ),
    }
    if not all(integrity_checks.values()):
        decision = "KILL_G3_E0_PREFLIGHT_INTEGRITY"
    elif all(readiness_checks.values()):
        decision = "READY_G3_E0_LABEL_FIT_EXECUTION"
    else:
        decision = "HOLD_G3_E0_COST_OR_COVERAGE"

    compact_tasks = [
        {key: value for key, value in task.items() if key != "record"}
        for task in tasks
    ]
    e0_records = payload_with_hash(
        {
            "version": "g3_e0_records_v2",
            "source_record_manifest_file_sha256":
                v1.V1_RECORD_FILE_SHA256,
            "source_record_manifest_payload_sha256":
                v1.V1_RECORD_PAYLOAD_SHA256,
            "records": records,
            "records_sha256": canonical_sha256(records),
            "label_values_opened": False,
            "models_fit": 0,
        }
    )
    e0_streams = payload_with_hash(
        {
            "version": "g3_e0_streams_v2",
            "source_stream_manifest_file_sha256":
                v1.V1_STREAM_FILE_SHA256,
            "source_stream_manifest_payload_sha256":
                v1.V1_STREAM_PAYLOAD_SHA256,
            "replicates": list(E0_REPLICATES),
            "rows": compact_tasks,
            "rows_sha256": canonical_sha256(compact_tasks),
            "streams_consumed": 0,
        }
    )
    by_record = {str(row["record_id"]): row for row in records}
    full_tasks = [
        {**task, "record": by_record[str(task["record_id"])]}
        for task in compact_tasks
    ]
    e0_tasks = payload_with_hash(
        {
            "version": "g3_e0_tasks_v2",
            "records_payload_sha256":
                e0_records["canonical_payload_sha256"],
            "streams_payload_sha256":
                e0_streams["canonical_payload_sha256"],
            "tasks": full_tasks,
            "tasks_sha256": canonical_sha256(full_tasks),
            "label_values_opened": False,
        }
    )
    lock = {
        "version": VERSION,
        "runner_version": RUNNER_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "terminal_status": "HOLD_G3_AFTER_E0_V2_PREFLIGHT_SEAL",
        "v1_authoritative_decision_preserved":
            "KILL_G3_E0_PREFLIGHT_INTEGRITY",
        "out_dir_resolved": str(out_dir.resolve()),
        "future_command": FUTURE_COMMAND,
        "jobs": 1,
        "nice": v1.MIN_NICE,
        "charter": {
            "base_path": str(CHARTER_PATH),
            "base_sha256": CHARTER_SHA256,
            "v2_amendment_path": str(AMENDMENT_PATH),
            "v2_amendment_sha256": AMENDMENT_SHA256,
        },
        "implementation": {
            "runner_path": "threes_rl/g3_e0_label_fit.py",
            "runner_sha256": RUNNER_SHA256,
            "preflight_path": str(IMPLEMENTATION_PATH),
            "preflight_sha256": sha256_path(IMPLEMENTATION_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_path(TEST_PATH),
            "test_evidence": tests,
        },
        "input_audit": inputs,
        "record_audit": record_audit,
        "task_audit": {
            "tasks": len(tasks),
            "by_partition": by_partition,
            "coupling": coupling,
            "collision": collision,
            "checks": task_checks,
            "passes": all(task_checks.values()),
        },
        "incumbent_artifact_audit": incumbent,
        "incumbent_policy_spec": incumbent["incumbent_policy_spec"],
        "transfer_preflight_lock_path":
            str(g3v1.TRANSFER_PREFLIGHT_LOCK_PATH),
        "cost_projection": stage_cost,
        "cost_checks": cost_checks,
        "disk": disk,
        "operations": operations,
        "integrity_checks": integrity_checks,
        "readiness_checks": readiness_checks,
        "ordinary_path_count":
            v1.EXPECTED_E0_BY_PARTITION["train"]
            + v1.EXPECTED_E0_BY_PARTITION["development"],
        "transfer_path_count":
            v1.EXPECTED_E0_BY_PARTITION["transfer_diagnostic"],
        "record_manifest_name": "E0_RECORD_MANIFEST.json",
        "stream_manifest_name": "E0_STREAM_MANIFEST.json",
        "task_manifest_name": "E0_TASK_MANIFEST.json",
        "zero_forbidden_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "label_paths_generated": 0,
            "label_values_opened": False,
            "scientific_models_fit": 0,
            "checkpoint_predictions": 0,
            "transfer_outcomes_opened": 0,
            "policy_outcomes": 0,
            "score_inspection": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
        "e0_execution_authorized":
            decision == "READY_G3_E0_LABEL_FIT_EXECUTION",
        "e1_authorized": False,
        "policy_evaluation_authorized": False,
        "promotion_authorized": False,
        "dashboard_eligible": False,
    }
    return lock, e0_records, e0_streams, e0_tasks


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    staging = out_dir.with_name(out_dir.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        lock, records, streams, tasks = build_preflight_payload(
            out_dir=out_dir
        )
        write_immutable_json(staging / "E0_RECORD_MANIFEST.json", records)
        write_immutable_json(staging / "E0_STREAM_MANIFEST.json", streams)
        write_immutable_json(staging / "E0_TASK_MANIFEST.json", tasks)
        lock["record_manifest_file_sha256"] = sha256_path(
            staging / "E0_RECORD_MANIFEST.json"
        )
        lock["stream_manifest_file_sha256"] = sha256_path(
            staging / "E0_STREAM_MANIFEST.json"
        )
        lock["task_manifest_file_sha256"] = sha256_path(
            staging / "E0_TASK_MANIFEST.json"
        )
        lock = payload_with_hash(lock)
        write_immutable_json(staging / "preflight_lock.json", lock)
        staging.replace(out_dir)
        return lock
    except Exception as error:
        failure = payload_with_hash(
            {
                "version": VERSION,
                "decision": "KILL_G3_E0_PREFLIGHT_INTEGRITY",
                "error": f"{type(error).__name__}: {error}",
                "zero_forbidden_work": {
                    "new_games": 0,
                    "streams_consumed": 0,
                    "label_paths_generated": 0,
                    "label_values_opened": False,
                    "scientific_models_fit": 0,
                    "transfer_outcomes_opened": 0,
                    "policy_outcomes": 0,
                    "score_inspection": False,
                },
            }
        )
        write_immutable_json(staging / "PREFLIGHT_FAILURE.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if os.nice(0) < v1.MIN_NICE:
        os.nice(v1.MIN_NICE - os.nice(0))
    result = run_preflight(args.out_dir)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "canonical_payload_sha256":
                    result["canonical_payload_sha256"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

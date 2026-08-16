"""Outcome-free readiness audit for the G3 scale-transfer bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from threes_rl import g1r_acquire as g1r
from threes_rl import g2_fresh_transfer_acquire as transfer_acquire
from threes_rl import g2_scale_relational_hazard_preflight as g2_preflight
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.g2_scale_relational_hazard import feature_vector, schema_sha256
from threes_rl.replay_provenance import (
    GENUINE_ROOT_ORIGINS,
    replay_provenance,
)
from threes_rl.restart_manifest import canonical_ancestry_id, state_signature
from threes_rl.s3_power_preflight import sha256_path
from threes_rl.sim import DIRECTION_NAMES, ThreesSim
from threes_rl.train_td import state_from_replay_payload


VERSION = "g3_scale_transfer_bootstrap_preflight_v1"
CHARTER_PATH = Path("threes_rl/G3_SCALE_TRANSFER_BOOTSTRAP_CHARTER.md")
AMENDMENT_PATH = Path(
    "threes_rl/G3_SCALE_TRANSFER_BOOTSTRAP_CHARTER_AMENDMENT_A1.md"
)
IMPLEMENTATION_PATH = Path(
    "threes_rl/g3_scale_transfer_bootstrap_preflight.py"
)
TEST_PATH = Path("tests/test_rl_g3_scale_transfer_bootstrap_preflight.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "g3_scale_transfer_bootstrap_preflight_v1_test_evidence.json"
)
OUTPUT_DIR = Path(
    "threes_rl/runs/forensics/g3_scale_transfer_bootstrap_preflight_v1"
)
ROOT_MANIFEST_PATH = Path(
    "threes_rl/runs/forensics/g2_scale_equivariant_relational_hazard/"
    "G2_ROOT_MANIFEST.json"
)
G2_PREFLIGHT_PATH = ROOT_MANIFEST_PATH.with_name("G2_PREFLIGHT.json")
TRANSFER_RESULT_PATH = Path(
    "threes_rl/runs/forensics/g2_fresh_transfer_acquisition_v1/"
    "G2_TRANSFER_ACQUISITION_RESULT.json"
)
TRANSFER_PREFLIGHT_LOCK_PATH = TRANSFER_RESULT_PATH.with_name(
    "preflight_lock.json"
)
TRANSFER_SOURCE_DIR = TRANSFER_RESULT_PATH.parent / "qualifying_sources"
INCUMBENT_PATH = Path("threes_rl/current_incumbent_policy.txt")
RUNS_ROOT = Path("threes_rl/runs")
RUNTIME_BENCHMARK_SUMMARY = Path(
    "threes_rl/runs/forensics/r15a_context_a2/natural_labels/summary.json"
)
RUNTIME_BENCHMARK_LABELS = Path(
    "threes_rl/runs/forensics/r15a_context_a2/natural_labels/labels.jsonl"
)
RUNTIME_BENCHMARK_MANIFEST = Path(
    "threes_rl/runs/forensics/r15a_context_a2/R15A_A2_LABEL_MANIFEST.json"
)

CHARTER_SHA256 = (
    "e216aa50737afee0d439e060cc9b1e1f24d2f552af4c3f0c8944470ff7a45fc1"
)
AMENDMENT_SHA256 = (
    "baba72003934ef55a48383704e4c6b5738787d561a0d81f0ea09383f64122b94"
)
ROOT_MANIFEST_FILE_SHA256 = (
    "60d514ed79ff315f7c2e0d2ad13bb712a57d4c3b204587691aa878a7486ea2ca"
)
ROOT_MANIFEST_PAYLOAD_SHA256 = (
    "15ecb9d52ae66e938952a07a8c3d6ef3f2d39b0dd1ef3ecb3c1e4e6fcab031ce"
)
G2_PREFLIGHT_FILE_SHA256 = (
    "2e1084f2a0673935866839e89765d3d1a31a2c2348e99c01edc9abc2405f05cc"
)
TRANSFER_RESULT_FILE_SHA256 = (
    "7b862377546b35c8c53967eedd39edb736c5db039d262f65048da4c47774ca74"
)
TRANSFER_RESULT_PAYLOAD_SHA256 = (
    "a464287ea64a9cac11971cbec9ba45731291c9bc9dacfdbe472ee6661895cee4"
)
TRANSFER_SOURCE_MANIFEST_SHA256 = (
    "e689accbd2f5f7a869112efb884de1a1ef80d78ab61a75183352749d8c7daba9"
)
SCHEMA_SHA256 = (
    "6af0cd515e5886b5fd8bc4d9f52cc9202bd3ed1f149d0ae146829681aea8340e"
)
INCUMBENT_FILE_SHA256 = (
    "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"
)

INPUT_FILE_LOCKS = {
    Path("threes_rl/G2_SCALE_EQUIVARIANT_RELATIONAL_HAZARD_PROPOSAL.md"):
        "43b413c1a8145a25750009cc3048bbda6127a44cfccbf72c7d1710e1e6027099",
    Path("threes_rl/g2_scale_relational_hazard.py"):
        "9ffaa45dd36b633cdae10110fdaefc8cd27053ab3f0216ddb3f1886ea625af8a",
    Path("threes_rl/g2_scale_relational_hazard_preflight.py"):
        "b5feebe5965258016480aca95f9a690392f0c3bdd7d0a3b73d5efddf35f02559",
    G2_PREFLIGHT_PATH: G2_PREFLIGHT_FILE_SHA256,
    ROOT_MANIFEST_PATH: ROOT_MANIFEST_FILE_SHA256,
    TRANSFER_RESULT_PATH: TRANSFER_RESULT_FILE_SHA256,
    Path("threes_rl/g2_fresh_transfer_acquire.py"):
        "66ce0dea164a2c34fe8cbf5e92d35e8797116ed83c7ec15500c03b02c7f87c23",
    Path("threes_rl/g2_transfer_acquisition_guard.py"):
        "0cd3f655d5aba9c6d7bbb9ba710a91bfedcdaeea7893868db6989b1e4ff40500",
    INCUMBENT_PATH: INCUMBENT_FILE_SHA256,
    Path("threes_rl/sim.py"):
        "67e7a245c05e59367402095ad018122fb4cb1ef08664bf28bf4bc03a02a73072",
    Path("threes_rl/eval.py"):
        "df0a558014583fcfd24fd8ddf48988e375ad9a6fc5199d35311c40d8b6a3f705",
    Path("threes_rl/expectimax.py"):
        "98a7f0d05437d01555ea37d21211fa36d7260cba84456b0fb08799472b26ec14",
    Path("threes_rl/split_eval.py"):
        "b71c66cb289b37437272568762f56fda8c82d468a4af0021fc9089dd2a05a8c2",
}

PARTITION_LOCKS = {
    "train": {
        "records": 550,
        "roots": 283,
        "records_sha256":
            "5858ed61befcd521d3f70ba496d2c7bf2782541e295e9d90bd23897dae77fceb",
        "record_ids_sha256":
            "cff3567780ab8fd21cd812c1ba7d3addd244bf25c0be941706b7d4401e716db9",
        "roots_sha256":
            "ea8b66cb91dcbffefcf03ca8c20cb1a0366ac8b146c252badefd2880f55fb55a",
    },
    "development": {
        "records": 133,
        "roots": 69,
        "records_sha256":
            "6a210761eaa832f6413516776a0237859eea7cc23987ecc57779d133f8470619",
        "record_ids_sha256":
            "a6ed4d1266d888c0f431e43823f99e9e45865df9980c0b424a76bbef291e36a8",
        "roots_sha256":
            "318da48e46ca93ed0efaae2b1ae30ea5ba520a4dd7ef23a7d25b760a8e713d8f",
    },
}

STREAM_BASES = {
    "logical_seed": 57_000_000_000,
    "deck_stream_id": 58_000_000_000,
    "slot_stream_id": 59_000_000_000,
    "policy_stream_id": 60_000_000_000,
}
REPLICATES = 8
HORIZONS = (10, 20, 40)
POWER_OR_GRID = (
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    3.00,
    4.00,
    5.00,
    6.00,
    8.00,
    10.00,
    15.00,
    20.00,
    30.00,
)
POWER_REQUIRED = 0.80
MAX_INCREMENTAL_BYTES = 4 * 1024**3
MAX_PROJECTED_RUNTIME_SECONDS = 72 * 60 * 60
MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
MIN_NICE = 10


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _compact_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "board": payload.get("board"),
        "preview": payload.get("preview"),
        "tile_cycle": payload.get("tile_cycle"),
        "move_count": payload.get("move_count"),
        "game_over": payload.get("game_over"),
        "legal_actions": payload.get("legal_actions"),
        "legal_mask": payload.get("legal_mask"),
    }


def _partition_rows(
    root_manifest: dict[str, Any], partition: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = sorted(
        [
            row
            for row in root_manifest["records"]
            if row.get("partition") == partition
        ],
        key=lambda row: (
            str(row["root_cluster"]),
            str(row["scale"]),
            str(row["record_id"]),
        ),
    )
    roots = sorted({str(row["root_cluster"]) for row in rows})
    summary = {
        "records": len(rows),
        "roots": len(roots),
        "records_sha256": canonical_sha256(rows),
        "record_ids_sha256": canonical_sha256(
            [str(row["record_id"]) for row in rows]
        ),
        "roots_sha256": canonical_sha256(roots),
        "records_by_scale": dict(
            sorted(Counter(str(row["scale"]) for row in rows).items())
        ),
        "roots_by_family": dict(
            sorted(
                Counter(
                    next(
                        str(row["behavior_family"])
                        for row in rows
                        if str(row["root_cluster"]) == root
                    )
                    for root in roots
                ).items()
            )
        ),
    }
    return rows, summary


def _find_frame(
    replay: dict[str, Any], source_frame_index: int
) -> dict[str, Any]:
    matches = [
        frame
        for fallback, frame in enumerate(replay.get("frames", []))
        if isinstance(frame, dict)
        and int(frame.get("index", fallback)) == source_frame_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one frame {source_frame_index}, found {len(matches)}"
        )
    payload = matches[0].get("state")
    if not isinstance(payload, dict):
        raise ValueError("Selected frame has no state")
    return payload


def _feature_totality(
    state_payload: dict[str, Any],
    *,
    starter_tile: int,
    target: int,
    legal_actions: list[int],
) -> str:
    state = state_from_replay_payload(state_payload)
    sim = ThreesSim.from_stream_ids(
        deck_stream_id=2_026_072_531,
        slot_stream_id=2_026_072_532,
        starter_tile=starter_tile,
    )
    rows = []
    for action in legal_actions:
        for horizon in HORIZONS:
            vector = feature_vector(
                state,
                sim,
                action,
                target=target,
                horizon=horizon,
                starter_tile=starter_tile,
            )
            if vector.shape != (64,) or not np.all(np.isfinite(vector)):
                raise ValueError("G2 feature schema is not finite width 64")
            if not np.all((vector >= 0.0) & (vector <= 1.0)):
                raise ValueError("G2 feature outside [0,1]")
            rows.append(vector.astype(np.float64, copy=False).tobytes())
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row)
    return digest.hexdigest()


def validate_ordinary_records(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    replay_cache: dict[str, dict[str, Any]] = {}
    replay_sha_cache: dict[str, str] = {}
    failures: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for row in rows:
        path_text = str(row["source_replay"])
        path = Path(path_text)
        try:
            if path_text not in replay_cache:
                replay_sha_cache[path_text] = sha256_path(path)
                replay_cache[path_text] = _json(path)
            replay = replay_cache[path_text]
            if replay_sha_cache[path_text] != row["source_replay_sha256"]:
                raise ValueError("source replay hash mismatch")
            provenance = replay_provenance(replay, path)
            if (
                provenance.get("replay_origin") not in GENUINE_ROOT_ORIGINS
                or provenance.get("root_origin") not in GENUINE_ROOT_ORIGINS
                or not provenance.get("replay_reset_invariant")
            ):
                raise ValueError("source replay is not a direct natural root")
            if canonical_ancestry_id(replay, path) != row["root_cluster"]:
                raise ValueError("canonical ancestry mismatch")
            starter = int(row.get("starter_tile") or 1536)
            payload = _find_frame(replay, int(row["source_frame_index"]))
            if _compact_state(payload) != row["state"]:
                raise ValueError("selected compact state mismatch")
            if state_signature(payload, starter) != row["state_sha1"]:
                raise ValueError("selected state hash mismatch")
            state = state_from_replay_payload(payload)
            validator = ThreesSim.from_stream_ids(
                deck_stream_id=2_026_072_533,
                slot_stream_id=2_026_072_534,
                starter_tile=starter,
            )
            legal = validator.legal_actions(state)
            legal_names = [DIRECTION_NAMES[action] for action in legal]
            if legal_names != row["state"]["legal_actions"] or not legal:
                raise ValueError("legal action mismatch")
            feature_sha = _feature_totality(
                payload,
                starter_tile=starter,
                target=int(row["target"]),
                legal_actions=legal,
            )
            validated.append(
                {
                    "partition": row["partition"],
                    "record_id": row["record_id"],
                    "root_cluster": row["root_cluster"],
                    "behavior_family": row["behavior_family"],
                    "scale": row["scale"],
                    "target": int(row["target"]),
                    "source_replay": path_text,
                    "source_replay_sha256": row["source_replay_sha256"],
                    "source_frame_index": int(row["source_frame_index"]),
                    "state_sha1": row["state_sha1"],
                    "starter_tile": starter,
                    "legal_actions": legal_names,
                    "legal_action_ids": legal,
                    "legal_action_count": len(legal),
                    "feature_rows_sha256": feature_sha,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "record_id": row.get("record_id"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return validated, {
        "records": len(rows),
        "validated_records": len(validated),
        "unique_sources": len(replay_cache),
        "failures": failures,
        "passes": not failures and len(validated) == len(rows),
    }


def _transfer_sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    source_audit = result.get("source_audit")
    if not isinstance(source_audit, dict):
        raise ValueError("Missing sealed transfer source audit")
    sources = source_audit.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Missing sealed transfer sources")
    if canonical_sha256(sources) != TRANSFER_SOURCE_MANIFEST_SHA256:
        raise ValueError("Transfer source manifest hash mismatch")
    return sources


def validate_transfer_records(
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sources):
        replay_path = Path(str(source["source_replay"]))
        state_path = Path(str(source["source_state"]))
        try:
            if not replay_path.is_relative_to(TRANSFER_SOURCE_DIR):
                raise ValueError("transfer replay escaped sealed source dir")
            if not state_path.is_relative_to(TRANSFER_SOURCE_DIR):
                raise ValueError("transfer state escaped sealed source dir")
            if sha256_path(replay_path) != source["source_replay_sha256"]:
                raise ValueError("transfer replay hash mismatch")
            if sha256_path(state_path) != source["source_state_sha256"]:
                raise ValueError("transfer state artifact hash mismatch")
            replay = _json(replay_path)
            state_record = _json(state_path)
            extracted = transfer_acquire.extract_first_transfer_state(
                replay,
                family=str(source["family"]),
                expected_seed=int(source["logical_seed"]),
            )
            if extracted is None:
                raise ValueError("transfer replay no longer qualifies")
            for key in (
                "root_cluster",
                "source_frame_index",
                "state_sha1",
            ):
                if extracted[key] != source[key]:
                    raise ValueError(f"transfer extracted {key} mismatch")
            if state_record["state_sha1"] != source["state_sha1"]:
                raise ValueError("transfer state record mismatch")
            payload = state_record.get("state")
            if not isinstance(payload, dict):
                raise ValueError("transfer state payload missing")
            starter = int(replay.get("starter_tile", 1536))
            if state_signature(payload, starter) != source["state_sha1"]:
                raise ValueError("transfer state signature mismatch")
            state = state_from_replay_payload(payload)
            validator = ThreesSim.from_stream_ids(
                deck_stream_id=2_026_072_535,
                slot_stream_id=2_026_072_536,
                starter_tile=starter,
            )
            legal = validator.legal_actions(state)
            legal_names = [DIRECTION_NAMES[action] for action in legal]
            if payload.get("legal_actions") != legal_names or not legal:
                raise ValueError("transfer legal action mismatch")
            feature_sha = _feature_totality(
                payload,
                starter_tile=starter,
                target=3072,
                legal_actions=legal,
            )
            validated.append(
                {
                    "partition": "transfer_diagnostic",
                    "record_id": f"g3-transfer-{ordinal:02d}",
                    "root_cluster": source["root_cluster"],
                    "behavior_family": source["family"],
                    "scale": "pre3072_transfer",
                    "target": 3072,
                    "source_replay": str(replay_path),
                    "source_replay_sha256": source["source_replay_sha256"],
                    "source_state": str(state_path),
                    "source_state_sha256": source["source_state_sha256"],
                    "source_frame_index": int(source["source_frame_index"]),
                    "state_sha1": source["state_sha1"],
                    "starter_tile": starter,
                    "legal_actions": legal_names,
                    "legal_action_ids": legal,
                    "legal_action_count": len(legal),
                    "feature_rows_sha256": feature_sha,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "source_replay": str(replay_path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return validated, {
        "records": len(sources),
        "validated_records": len(validated),
        "unique_roots": len({row["root_cluster"] for row in validated}),
        "roots_by_family": dict(
            sorted(Counter(row["behavior_family"] for row in validated).items())
        ),
        "failures": failures,
        "passes": (
            not failures
            and len(validated) == 32
            and len({row["root_cluster"] for row in validated}) == 32
        ),
    }


def _rg_matching_paths(patterns: Iterable[str], search_root: Path) -> list[str]:
    unique = sorted({str(pattern) for pattern in patterns if str(pattern)})
    if not unique:
        return []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as handle:
        handle.write("\n".join(unique))
        handle.write("\n")
        handle.flush()
        result = subprocess.run(
            [
                "rg",
                "-l",
                "-F",
                "-f",
                handle.name,
                "--glob",
                "!forensics/g2_fresh_transfer_acquisition_v1/**",
                "--glob",
                "!forensics/g3_scale_transfer_bootstrap_preflight_v1/**",
                str(search_root),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"rg metadata audit failed: {result.stderr.strip()}")
    return sorted(line for line in result.stdout.splitlines() if line)


def transfer_untouched_audit(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    root_matches = _rg_matching_paths(
        [row["root_cluster"] for row in records], RUNS_ROOT
    )
    state_matches = _rg_matching_paths(
        [row["state_sha1"] for row in records], RUNS_ROOT
    )
    matches = sorted(set(root_matches) | set(state_matches))
    return {
        "root_token_matches_outside_source": root_matches,
        "state_token_matches_outside_source": state_matches,
        "matched_paths": matches,
        "matched_path_hashes": {
            path: sha256_path(Path(path))
            for path in matches
            if Path(path).is_file()
        },
        "passes": not matches,
    }


def _legacy_candidate_paths() -> dict[str, Any]:
    file_matches = _rg_matching_paths(
        [ROOT_MANIFEST_FILE_SHA256], RUNS_ROOT
    )
    payload_matches = _rg_matching_paths(
        [ROOT_MANIFEST_PAYLOAD_SHA256], RUNS_ROOT
    )
    both = sorted(set(file_matches).intersection(payload_matches))
    excluded_inputs = {
        str(ROOT_MANIFEST_PATH),
        str(G2_PREFLIGHT_PATH),
        str(TRANSFER_PREFLIGHT_LOCK_PATH),
    }
    candidates = [path for path in both if path not in excluded_inputs]
    return {
        "root_manifest_file_hash_matches": file_matches,
        "root_manifest_payload_hash_matches": payload_matches,
        "both_hashes_matches": both,
        "excluded_immutable_inputs": sorted(excluded_inputs.intersection(both)),
        "candidate_metadata_paths": candidates,
    }


def _compatible_legacy_paths(
    candidate_paths: list[str],
    required_by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[set[tuple[str, str, int]], list[dict[str, Any]]]:
    covered: set[tuple[str, str, int]] = set()
    reports: list[dict[str, Any]] = []
    for text in candidate_paths:
        path = Path(text)
        reasons: list[str] = []
        try:
            payload = _json(path)
            contract = payload.get("g3_compatible_label_contract")
            if not isinstance(contract, dict):
                reasons.append("missing_g3_compatible_label_contract")
            else:
                checks = {
                    "root_manifest_file_sha256":
                        ROOT_MANIFEST_FILE_SHA256,
                    "root_manifest_payload_sha256":
                        ROOT_MANIFEST_PAYLOAD_SHA256,
                    "incumbent_policy_file_sha256":
                        INCUMBENT_FILE_SHA256,
                    "continuation_policy": "frozen_incumbent_depth2",
                    "horizons_from_one_h40_path": True,
                    "terminal_right_censoring": True,
                    "all_legal_actions": True,
                    "replicates": REPLICATES,
                    "shared_action_arm_tapes": True,
                }
                for key, expected in checks.items():
                    if contract.get(key) != expected:
                        reasons.append(f"contract_mismatch:{key}")
                rows = contract.get("path_provenance")
                if not isinstance(rows, list):
                    reasons.append("missing_path_provenance")
                elif not reasons:
                    local: set[tuple[str, str, int]] = set()
                    for row in rows:
                        key = (
                            str(row.get("record_id")),
                            str(row.get("action")),
                            int(row.get("replicate", -1)),
                        )
                        required = required_by_key.get(key)
                        if required is None:
                            continue
                        provenance_keys = (
                            "root_cluster",
                            "state_sha1",
                            "logical_seed",
                            "deck_stream_id",
                            "slot_stream_id",
                            "policy_stream_id",
                        )
                        if all(
                            row.get(item) == required.get(item)
                            for item in provenance_keys
                        ):
                            local.add(key)
                    covered.update(local)
            reports.append(
                {
                    "path": text,
                    "sha256": sha256_path(path),
                    "compatible": not reasons,
                    "reasons": reasons,
                }
            )
        except Exception as error:
            reports.append(
                {
                    "path": text,
                    "sha256": sha256_path(path) if path.is_file() else None,
                    "compatible": False,
                    "reasons": [f"{type(error).__name__}: {error}"],
                }
            )
    return covered, reports


def label_stream_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_ordinal, record in enumerate(records):
        for action_id, action_name in zip(
            record["legal_action_ids"], record["legal_actions"]
        ):
            for replicate in range(REPLICATES):
                streams = {
                    key: base + 8 * record_ordinal + replicate
                    for key, base in STREAM_BASES.items()
                }
                rows.append(
                    {
                        "record_ordinal": record_ordinal,
                        "partition": record["partition"],
                        "record_id": record["record_id"],
                        "root_cluster": record["root_cluster"],
                        "behavior_family": record["behavior_family"],
                        "scale": record["scale"],
                        "target": int(record["target"]),
                        "state_sha1": record["state_sha1"],
                        "action_id": int(action_id),
                        "action": action_name,
                        "replicate": replicate,
                        "block": "A" if replicate < 4 else "B",
                        **streams,
                    }
                )
    return rows


def stream_coupling_audit(
    rows: list[dict[str, Any]], *, exclude_dir: Path
) -> dict[str, Any]:
    by_unit: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[(row["record_ordinal"], row["replicate"])].append(row)
    coupling_pass = all(
        all(
            len({int(row[key]) for row in unit_rows}) == 1
            for key in STREAM_BASES
        )
        for unit_rows in by_unit.values()
    )
    unit_streams = {
        (
            key,
            int(unit_rows[0][key]),
        )
        for unit_rows in by_unit.values()
        for key in STREAM_BASES
    }
    expected_unit_streams = len(by_unit) * len(STREAM_BASES)
    prior, sources = g1r.historical_collision_union(exclude_dir=exclude_dir)
    collisions: dict[str, list[int]] = {}
    for key in STREAM_BASES:
        prior_values = set(prior.get(key, set()))
        if key == "logical_seed":
            for alias in ("seed", "root_seed", "source_seed", "fresh_root_seed"):
                prior_values.update(prior.get(alias, set()))
        requested = {
            int(unit_rows[0][key]) for unit_rows in by_unit.values()
        }
        collisions[key] = sorted(requested.intersection(prior_values))
    checks = {
        "shared_within_action_arms": coupling_pass,
        "unique_across_units_and_kinds":
            len(unit_streams) == expected_unit_streams,
        "historical_collisions_zero": not any(collisions.values()),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "record_replicate_units": len(by_unit),
        "intended_action_arm_reuses": len(rows) - len(by_unit),
        "unique_stream_ids_across_kinds": len(unit_streams),
        "collisions": collisions,
        "historical_union": sources,
    }


def label_coverage_inventory(
    records: list[dict[str, Any]],
    stream_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_by_key = {
        (row["record_id"], row["action"], row["replicate"]): row
        for row in stream_rows
    }
    legacy_search = _legacy_candidate_paths()
    covered, reports = _compatible_legacy_paths(
        legacy_search["candidate_metadata_paths"], required_by_key
    )
    missing = sorted(set(required_by_key).difference(covered))
    required_by_partition = Counter(row["partition"] for row in stream_rows)
    compatible_by_partition = Counter(
        required_by_key[key]["partition"] for key in covered
    )
    return {
        "compatibility_rule":
            "exact machine-readable G3 sidecar contract; values unopened",
        "metadata_search": legacy_search,
        "candidate_reports": reports,
        "required_h40_paths": len(required_by_key),
        "compatible_existing_h40_paths": len(covered),
        "missing_h40_paths": len(missing),
        "maximum_interval_rows": len(required_by_key) * len(HORIZONS),
        "required_by_partition": dict(sorted(required_by_partition.items())),
        "compatible_by_partition": dict(
            sorted(compatible_by_partition.items())
        ),
        "missing_by_partition": {
            partition: required_by_partition[partition]
            - compatible_by_partition[partition]
            for partition in sorted(required_by_partition)
        },
        "required_path_key_sha256": canonical_sha256(
            sorted(
                [
                    [record_id, action, replicate]
                    for record_id, action, replicate in required_by_key
                ]
            )
        ),
        "compatible_path_key_sha256": canonical_sha256(
            sorted(
                [
                    [record_id, action, replicate]
                    for record_id, action, replicate in covered
                ]
            )
        ),
        "missing_path_key_sha256": canonical_sha256(
            [
                [record_id, action, replicate]
                for record_id, action, replicate in missing
            ]
        ),
        "label_values_opened": False,
    }


def power_audit_n32() -> dict[str, Any]:
    rows = []
    for odds_ratio in POWER_OR_GRID:
        try:
            row = g2_preflight.simulate_power(32, odds_ratio)
            row["attainable"] = True
        except ValueError as error:
            row = {
                "roots": 32,
                "target_policy_odds_ratio": odds_ratio,
                "attainable": False,
                "reason": str(error),
                "power_ci_above_zero": None,
                "power_pass_point_or_1_25_and_ci": None,
            }
        rows.append(row)
    passing = [
        row["target_policy_odds_ratio"]
        for row in rows
        if row["attainable"]
        and row["power_pass_point_or_1_25_and_ci"] >= POWER_REQUIRED
    ]
    return {
        "roots": 32,
        "assumptions": {
            "base_rate": g2_preflight.POWER_BASE_RATE,
            "root_rho": g2_preflight.POWER_ROOT_RHO,
            "repeats": g2_preflight.POWER_REPEATS,
            "activity": g2_preflight.POWER_ACTIVITY,
            "draws": g2_preflight.POWER_DRAWS,
            "required_power": POWER_REQUIRED,
            "or_grid": list(POWER_OR_GRID),
            "simulation_seed_formula":
                "2026072508 + 32*100 + round(100*OR)",
        },
        "rows": rows,
        "mde_grid_or": min(passing) if passing else None,
        "informative_only_at_or_above_mde": True,
    }


def cost_projection(missing_paths: int) -> dict[str, Any]:
    summary = _json(RUNTIME_BENCHMARK_SUMMARY)
    elapsed = float(summary["elapsed_s_this_invocation"])
    jobs = int(summary["jobs"])
    new_tasks = int(summary["new_tasks"])
    measured_cpu_seconds_per_path = elapsed * jobs / new_tasks
    conservative_seconds_per_path = 1.25 * measured_cpu_seconds_per_path
    runtime_seconds = missing_paths * conservative_seconds_per_path
    benchmark_bytes = (
        RUNTIME_BENCHMARK_LABELS.stat().st_size
        + RUNTIME_BENCHMARK_MANIFEST.stat().st_size
    )
    measured_bytes_per_path = benchmark_bytes / new_tasks
    bytes_per_path = max(4096, math.ceil(1.25 * measured_bytes_per_path))
    base_bytes = 16 * 1024**2
    projected_bytes = math.ceil(1.25 * (base_bytes + missing_paths * bytes_per_path))
    return {
        "benchmark": {
            "summary": str(RUNTIME_BENCHMARK_SUMMARY),
            "summary_sha256": sha256_path(RUNTIME_BENCHMARK_SUMMARY),
            "labels": str(RUNTIME_BENCHMARK_LABELS),
            "labels_bytes": RUNTIME_BENCHMARK_LABELS.stat().st_size,
            "manifest": str(RUNTIME_BENCHMARK_MANIFEST),
            "manifest_bytes": RUNTIME_BENCHMARK_MANIFEST.stat().st_size,
            "elapsed_seconds": elapsed,
            "jobs": jobs,
            "new_tasks": new_tasks,
            "measured_cpu_seconds_per_path": measured_cpu_seconds_per_path,
            "measured_bytes_per_path": measured_bytes_per_path,
        },
        "missing_h40_paths": missing_paths,
        "one_worker_nice": MIN_NICE,
        "runtime_multiplier": 1.25,
        "conservative_seconds_per_path": conservative_seconds_per_path,
        "projected_runtime_seconds": runtime_seconds,
        "projected_runtime_hours": runtime_seconds / 3600.0,
        "runtime_limit_seconds": MAX_PROJECTED_RUNTIME_SECONDS,
        "bytes_per_path_floor": 4096,
        "selected_bytes_per_path": bytes_per_path,
        "base_bytes": base_bytes,
        "storage_multiplier": 1.25,
        "projected_incremental_bytes": projected_bytes,
        "projected_incremental_gib": projected_bytes / 1024**3,
        "storage_limit_bytes": MAX_INCREMENTAL_BYTES,
        "runtime_passes": runtime_seconds <= MAX_PROJECTED_RUNTIME_SECONDS,
        "storage_passes": projected_bytes < MAX_INCREMENTAL_BYTES,
    }


def _verify_incumbent_artifacts(
    transfer_lock: dict[str, Any],
) -> dict[str, Any]:
    policy_lock = transfer_lock["policy_lock"]
    incumbent = next(
        row
        for row in policy_lock["families"]
        if row["family"] == "g2_transfer_phaseblend_incumbent"
    )
    failures = []
    checked_files = 0
    checked_bytes = 0
    for manifest in incumbent["checkpoint_manifests"]:
        for row in manifest["files"]:
            path = Path(row["path"])
            checked_files += 1
            try:
                stat = path.stat()
                checked_bytes += stat.st_size
                if stat.st_size != int(row["byte_size"]):
                    raise ValueError("byte size changed")
                if sha256_path(path) != row["sha256"]:
                    raise ValueError("sha256 changed")
            except Exception as error:
                failures.append(
                    {
                        "path": str(path),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    checks = {
        "incumbent_file_hash_exact":
            sha256_path(INCUMBENT_PATH) == INCUMBENT_FILE_SHA256,
        "policy_lock_self_hash_exact":
            canonical_sha256(
                {
                    key: value
                    for key, value in policy_lock.items()
                    if key != "policy_lock_sha256"
                }
            )
            == policy_lock["policy_lock_sha256"],
        "checkpoint_payloads_exact": not failures,
    }
    return {
        "policy_lock_sha256": policy_lock["policy_lock_sha256"],
        "incumbent_policy_spec": incumbent["policy_spec"],
        "incumbent_policy_spec_sha256": incumbent["policy_spec_sha256"],
        "checked_files": checked_files,
        "checked_bytes": checked_bytes,
        "failures": failures,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _disk_audit(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    return {
        "free_bytes": usage.free,
        "free_gib": free_gib,
        "minimum_free_gib": MIN_FREE_GIB,
        "target_free_gib": TARGET_FREE_GIB,
        "above_hard_minimum": free_gib >= MIN_FREE_GIB,
        "above_target": free_gib >= TARGET_FREE_GIB,
    }


def _input_hash_audit() -> dict[str, Any]:
    rows = []
    for path, expected in INPUT_FILE_LOCKS.items():
        actual = sha256_path(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    charter_actual = sha256_path(CHARTER_PATH)
    amendment_actual = sha256_path(AMENDMENT_PATH)
    rows.extend(
        [
            {
                "path": str(CHARTER_PATH),
                "expected_sha256": CHARTER_SHA256,
                "actual_sha256": charter_actual,
                "matches": charter_actual == CHARTER_SHA256,
            },
            {
                "path": str(AMENDMENT_PATH),
                "expected_sha256": AMENDMENT_SHA256,
                "actual_sha256": amendment_actual,
                "matches": amendment_actual == AMENDMENT_SHA256,
            },
        ]
    )
    return {"rows": rows, "passes": all(row["matches"] for row in rows)}


def _staging_path(out_dir: Path) -> Path:
    return out_dir.with_name(
        f"{out_dir.name}.staging.{os.getpid()}.{time.time_ns()}"
    )


def build_preflight_payload() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    input_hashes = _input_hash_audit()
    if not input_hashes["passes"]:
        raise ValueError("Immutable G3/G2 input hash mismatch")
    if schema_sha256() != SCHEMA_SHA256:
        raise ValueError("G2 feature schema hash mismatch")

    root_manifest = _json(ROOT_MANIFEST_PATH)
    if (
        root_manifest.get("canonical_payload_sha256")
        != ROOT_MANIFEST_PAYLOAD_SHA256
    ):
        raise ValueError("G2 root manifest payload hash mismatch")
    partition_rows: dict[str, list[dict[str, Any]]] = {}
    partition_summaries: dict[str, dict[str, Any]] = {}
    for partition in ("train", "development"):
        rows, summary = _partition_rows(root_manifest, partition)
        partition_rows[partition] = rows
        partition_summaries[partition] = summary
        if any(
            summary[key] != expected
            for key, expected in PARTITION_LOCKS[partition].items()
        ):
            raise ValueError(f"Frozen {partition} partition mismatch")

    ordinary_source_rows = (
        partition_rows["train"] + partition_rows["development"]
    )
    ordinary, ordinary_audit = validate_ordinary_records(ordinary_source_rows)

    transfer_result = _json(TRANSFER_RESULT_PATH)
    if (
        transfer_result.get("result_payload_sha256")
        != TRANSFER_RESULT_PAYLOAD_SHA256
    ):
        raise ValueError("Transfer result payload mismatch")
    transfer_sources = _transfer_sources(transfer_result)
    transfer, transfer_audit = validate_transfer_records(transfer_sources)

    ordinary_roots = {row["root_cluster"] for row in ordinary}
    transfer_roots = {row["root_cluster"] for row in transfer}
    root_overlap = sorted(ordinary_roots.intersection(transfer_roots))
    transfer_untouched = transfer_untouched_audit(transfer)

    all_records = ordinary + transfer
    stream_rows = label_stream_rows(all_records)
    stream_audit = stream_coupling_audit(
        stream_rows, exclude_dir=OUTPUT_DIR
    )
    coverage = label_coverage_inventory(all_records, stream_rows)
    power = power_audit_n32()
    cost = cost_projection(coverage["missing_h40_paths"])
    disk = _disk_audit(Path("."))
    services = g1r.service_health()
    heavy = _heavy_process_audit()
    current_nice = os.nice(0)
    transfer_lock = _json(TRANSFER_PREFLIGHT_LOCK_PATH)
    incumbent_artifacts = _verify_incumbent_artifacts(transfer_lock)

    test_evidence = _json(TEST_EVIDENCE_PATH)
    test_evidence_sha = sha256_path(TEST_EVIDENCE_PATH)
    tests_pass = bool(test_evidence.get("passes"))

    integrity_checks = {
        "immutable_input_hashes": input_hashes["passes"],
        "schema_hash_exact": schema_sha256() == SCHEMA_SHA256,
        "partition_hashes_exact": all(
            all(
                partition_summaries[partition][key] == expected
                for key, expected in PARTITION_LOCKS[partition].items()
            )
            for partition in PARTITION_LOCKS
        ),
        "ordinary_sources_exact": ordinary_audit["passes"],
        "transfer_sources_exact": transfer_audit["passes"],
        "ordinary_transfer_root_overlap_zero": not root_overlap,
        "transfer_panel_untouched": transfer_untouched["passes"],
        "stream_coupling_and_collisions_pass": stream_audit["passes"],
        "incumbent_payloads_exact": incumbent_artifacts["passes"],
        "focused_and_regression_tests_pass": tests_pass,
    }
    readiness_checks = {
        "all_required_paths_manifested":
            len(stream_rows) == coverage["required_h40_paths"],
        "coverage_inventory_complete":
            coverage["compatible_existing_h40_paths"]
            + coverage["missing_h40_paths"]
            == coverage["required_h40_paths"],
        "projected_runtime_within_72h": cost["runtime_passes"],
        "projected_incremental_storage_below_4gib":
            cost["storage_passes"],
        "disk_above_100gib": disk["above_hard_minimum"],
        "services_dashboard_top_three_pass": services["passes"],
        "no_competing_heavy_process": heavy["passes"],
        "nice_at_least_10": current_nice >= MIN_NICE,
        "n32_mde_computed": power["mde_grid_or"] is not None,
    }
    if not all(integrity_checks.values()):
        decision = "KILL_G3_PREFLIGHT_INTEGRITY"
    elif all(readiness_checks.values()):
        decision = "READY_G3_BOOTSTRAP_LABELS"
    else:
        decision = "HOLD_G3_LABEL_COVERAGE_OR_COST"

    record_manifest = {
        "version": "g3_scale_transfer_bootstrap_records_v1",
        "charter_sha256": CHARTER_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "root_manifest_file_sha256": ROOT_MANIFEST_FILE_SHA256,
        "root_manifest_payload_sha256": ROOT_MANIFEST_PAYLOAD_SHA256,
        "partition_summaries": partition_summaries,
        "records": all_records,
        "records_sha256": canonical_sha256(all_records),
        "score_or_label_outcome_opened": False,
    }
    record_manifest["canonical_payload_sha256"] = canonical_sha256(
        record_manifest
    )
    stream_manifest = {
        "version": "g3_scale_transfer_bootstrap_streams_v1",
        "charter_sha256": CHARTER_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "stream_bases": STREAM_BASES,
        "replicates": REPLICATES,
        "horizons": list(HORIZONS),
        "coupling":
            "same record/replicate tapes shared across every legal action",
        "rows": stream_rows,
        "rows_sha256": canonical_sha256(stream_rows),
        "streams_consumed": 0,
    }
    stream_manifest["canonical_payload_sha256"] = canonical_sha256(
        stream_manifest
    )
    payload = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "terminal_status": "HOLD_G3_AFTER_BOOTSTRAP_PREFLIGHT_SEAL",
        "charter": {
            "path": str(CHARTER_PATH),
            "file_sha256": CHARTER_SHA256,
            "amendment_path": str(AMENDMENT_PATH),
            "amendment_file_sha256": AMENDMENT_SHA256,
        },
        "implementation": {
            "path": str(IMPLEMENTATION_PATH),
            "sha256": sha256_path(IMPLEMENTATION_PATH),
            "test_path": str(TEST_PATH),
            "test_sha256": sha256_path(TEST_PATH),
            "test_evidence_path": str(TEST_EVIDENCE_PATH),
            "test_evidence_sha256": test_evidence_sha,
        },
        "input_hash_audit": input_hashes,
        "partition_summaries": partition_summaries,
        "ordinary_source_audit": ordinary_audit,
        "transfer_source_audit": transfer_audit,
        "ordinary_transfer_root_overlap": root_overlap,
        "transfer_untouched_audit": transfer_untouched,
        "record_manifest_canonical_payload_sha256":
            record_manifest["canonical_payload_sha256"],
        "stream_manifest_canonical_payload_sha256":
            stream_manifest["canonical_payload_sha256"],
        "stream_audit": stream_audit,
        "label_coverage": coverage,
        "n32_power_mde": power,
        "cost_projection": cost,
        "incumbent_artifact_audit": incumbent_artifacts,
        "integrity_checks": integrity_checks,
        "readiness_checks": readiness_checks,
        "disk": disk,
        "services": services,
        "heavy_process_audit": heavy,
        "current_nice": current_nice,
        "zero_forbidden_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "new_labels": 0,
            "label_values_opened": False,
            "models_fit": 0,
            "transfer_outcomes_opened": 0,
            "candidate_actions": 0,
            "rerankers_built": 0,
            "policy_outcomes": 0,
            "continuations": 0,
            "score_inspection": False,
            "incumbent_changed": False,
            "dashboard_changed": False,
        },
        "labels_authorized": False,
        "fitting_authorized": False,
        "policy_evaluation_authorized": False,
        "promotion_authorized": False,
        "dashboard_eligible": False,
    }
    payload["canonical_payload_sha256"] = canonical_sha256(payload)
    return payload, record_manifest, stream_manifest


def run_preflight(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    staging = _staging_path(out_dir)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        payload, records, streams = build_preflight_payload()
        _write_immutable_json(staging / "G3_RECORD_MANIFEST.json", records)
        _write_immutable_json(staging / "G3_LABEL_STREAM_MANIFEST.json", streams)
        payload["record_manifest_file_sha256"] = sha256_path(
            staging / "G3_RECORD_MANIFEST.json"
        )
        payload["stream_manifest_file_sha256"] = sha256_path(
            staging / "G3_LABEL_STREAM_MANIFEST.json"
        )
        payload["canonical_payload_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "canonical_payload_sha256"
            }
        )
        _write_immutable_json(staging / "G3_BOOTSTRAP_PREFLIGHT.json", payload)
        staging.replace(out_dir)
        return payload
    except Exception as error:
        failure = {
            "version": VERSION,
            "decision": "KILL_G3_PREFLIGHT_INTEGRITY",
            "error": f"{type(error).__name__}: {error}",
            "zero_forbidden_work": {
                "new_games": 0,
                "streams_consumed": 0,
                "new_labels": 0,
                "label_values_opened": False,
                "models_fit": 0,
                "transfer_outcomes_opened": 0,
                "policy_outcomes": 0,
                "score_inspection": False,
            },
        }
        failure["canonical_payload_sha256"] = canonical_sha256(failure)
        _write_immutable_json(staging / "PREFLIGHT_FAILURE.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    args = parser.parse_args()
    if os.nice(0) < MIN_NICE:
        os.nice(MIN_NICE - os.nice(0))
    payload = run_preflight(args.out_dir)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "canonical_payload_sha256":
                    payload["canonical_payload_sha256"],
                "out_dir": str(args.out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Outcome-free feasibility audit for a one-state-per-root K2 design."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from threes_rl import k1_engineering as k1
from threes_rl.eval import make_policy
from threes_rl.expectimax import NtupleExpectimaxPolicy
from threes_rl.g1r_qd_admission_v2 import _heavy_process_audit
from threes_rl.replay_provenance import initial_reset_diagnostics


VERSION = "k1_support_audit_v1"
ROOT = Path("threes_rl/runs/forensics/k1_compiled_kernel_v1")
OUTPUT_DIR = Path("threes_rl/runs/forensics/k1_support_audit_v1")
CHARTER_PATH = Path("threes_rl/K1_SUPPORT_AUDIT_CHARTER.md")
RUNNER_PATH = Path("threes_rl/k1_support_audit.py")
TEST_PATH = Path("tests/test_rl_k1_support_audit.py")
G1R_SIGNATURE_LOCK = Path(
    "threes_rl/runs/forensics/g1r_acquisition/pilot_v1/"
    "preflight_lock_pilot_v1.json"
)
QD_ADMISSION = Path(
    "threes_rl/runs/forensics/"
    "g1r_qd_admission_v2_terminal_schema/admission_result.json"
)
G1R_QD5_SEAL = Path(
    "threes_rl/runs/forensics/g1r_acquisition/"
    "pilot_v2_qd5/PILOT_V2_SEAL.json"
)
C2_TERMINAL = Path(
    "threes_rl/runs/forensics/c2_cost_admission_v1/"
    "C2_TERMINAL_RESULT.json"
)
EXPECTED_C2_TERMINAL_SHA256 = (
    "ac1e3b490a6ab7d498cacfdd1157ce68020ebe8459e7b654ac487fa28eb3cb9f"
)
EXPECTED_K1_TERMINAL_SHA256 = (
    "157d73f702705b6162af63bf0af98ca2fd404cf0e8fb76cc6ca5e72b3134e7a6"
)
EXPECTED_TOP_THREE = (263670, 261369, 258561)
MIN_ROOTS_PER_FAMILY = 12
PREFERRED_ROOTS_PER_FAMILY = 16


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("canonical_payload_sha256", None)
    result["canonical_payload_sha256"] = canonical_json_hash(result)
    return result


def verify_payload_hash(payload: Mapping[str, Any]) -> bool:
    raw = dict(payload)
    expected = raw.pop("canonical_payload_sha256", None)
    return isinstance(expected, str) and canonical_json_hash(raw) == expected


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def file_manifest(path: Path) -> dict[str, Any]:
    files = [path] if path.is_file() else [
        child for child in sorted(path.rglob("*")) if child.is_file()
    ]
    if not files:
        raise ValueError(f"Empty K1 support source: {path}")
    rows = [{
        "path": str(child),
        "relative_path":
            child.name if path.is_file() else str(child.relative_to(path)),
        "byte_size": child.stat().st_size,
        "sha256": sha256_path(child),
    } for child in files]
    return {
        "path": str(path),
        "file_count": len(rows),
        "total_bytes": sum(int(row["byte_size"]) for row in rows),
        "files": rows,
        "manifest_sha256": canonical_json_hash(rows),
    }


def _load_json(path: Path, *, require_payload_hash: bool = False) -> dict:
    payload = json.loads(path.read_text())
    if require_payload_hash and not verify_payload_hash(payload):
        raise ValueError(f"Payload hash mismatch: {path}")
    return payload


def _completion_rows() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (ROOT / "completed_games.jsonl").read_text().splitlines()
        if line
    ]
    keys = [
        (str(row["behavior_family"]), int(row["game_index"])) for row in rows
    ]
    if len(rows) != 108 or len(keys) != len(set(keys)):
        raise ValueError("K1 completion rows are not exactly 108 unique games")
    return rows


def _candidate_key(root: str, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (
            "K2-root-state-v1|"
            f"{root}|{int(row['frame_index'])}|{row['state_sha256']}"
        ).encode()
    ).hexdigest()


def qualifying_states(
    replay: Mapping[str, Any],
    *,
    root: str,
    incumbent: NtupleExpectimaxPolicy,
) -> list[dict[str, Any]]:
    candidates = [
        candidate
        for frame in replay.get("frames", [])
        if (candidate := k1._frame_candidate(frame, root=root)) is not None
    ]
    low_empty = [
        row for row in candidates
        if int(row["empty_count"]) <= k1.EMPTY_TRIGGER
    ]
    high_empty = sorted(
        (
            row for row in candidates
            if int(row["empty_count"]) > k1.EMPTY_TRIGGER
        ),
        key=lambda row: (
            hashlib.sha256(
                (
                    "K1-high-empty-v1|"
                    f"{root}|{row['frame_index']}|{row['state_sha256']}"
                ).encode()
            ).hexdigest(),
            int(row["frame_index"]),
        ),
    )[:8]
    qualifying = [
        {
            "root_ancestry": root,
            "frame_index": int(row["frame_index"]),
            "state_sha256": str(row["state_sha256"]),
        }
        for row in low_empty
    ]
    for row in high_empty:
        trigger = bool(
            k1._incumbent_metadata(row, incumbent)["trigger_reasons"][
                "low_margin"
            ]
        )
        if trigger:
            qualifying.append({
                "root_ancestry": root,
                "frame_index": int(row["frame_index"]),
                "state_sha256": str(row["state_sha256"]),
            })
    unique = {
        (int(row["frame_index"]), str(row["state_sha256"])): row
        for row in qualifying
    }
    return sorted(
        unique.values(),
        key=lambda row: (int(row["frame_index"]), str(row["state_sha256"])),
    )


def summarize_family_support(
    *,
    completion_rows: Sequence[Mapping[str, Any]],
    retained_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = {}
    families = [family for family, _spec in k1.FAMILY_SLATE]
    for family in families:
        all_family = [
            row for row in completion_rows
            if row["behavior_family"] == family
        ]
        retained = [
            row for row in retained_rows
            if row["behavior_family"] == family
        ]
        counts = [int(row["qualifying_state_count"]) for row in retained]
        observed = {
            f"ge_{threshold}": sum(count >= threshold for count in counts)
            for threshold in range(1, 5)
        }
        unobservable = len(all_family) - len(retained)
        summary[family] = {
            "attempted_roots": len(all_family),
            "completed_roots": sum(
                bool(row["complete"]) for row in all_family
            ),
            "retained_observable_roots": len(retained),
            "unobservable_roots": unobservable,
            "observed_roots_by_threshold": observed,
            "qualifying_state_count_distribution": dict(sorted(Counter(
                counts
            ).items())),
            "observed_ge1_lower_bound": observed["ge_1"],
            "ge1_maximum_possible_with_unknowns":
                observed["ge_1"] + unobservable,
            "meets_minimum_12_observed": observed["ge_1"]
                >= MIN_ROOTS_PER_FAMILY,
            "meets_preferred_16_observed": observed["ge_1"]
                >= PREFERRED_ROOTS_PER_FAMILY,
        }
    return summary


def root_diverse_decision(
    family_support: Mapping[str, Mapping[str, Any]],
    *,
    signature_families: int,
    integrity_passes: bool,
    alternative_trigger_support: Mapping[str, int],
) -> tuple[str, dict[str, Any]]:
    observed = {
        family: int(row["observed_ge1_lower_bound"])
        for family, row in family_support.items()
    }
    total = sum(observed.values())
    max_share = max(observed.values(), default=0) / max(total, 1)
    current_support = (
        len(observed) >= 3
        and all(count >= MIN_ROOTS_PER_FAMILY for count in observed.values())
        and max_share <= 0.40
    )
    supported_alternatives = [
        family for family, count in alternative_trigger_support.items()
        if int(count) >= MIN_ROOTS_PER_FAMILY
    ]
    alternative_ready = len(supported_alternatives) >= 3
    checks = {
        "integrity": integrity_passes,
        "at_least_three_signature_families": signature_families >= 3,
        "current_slate_minimum_12_each": current_support,
        "current_slate_max_family_share_le_40pct": max_share <= 0.40,
        "three_supported_alternative_families": alternative_ready,
        "no_unobservable_root_imputation": True,
    }
    ready = (
        checks["integrity"]
        and checks["at_least_three_signature_families"]
        and (current_support or alternative_ready)
    )
    return (
        "READY_K2_ROOT_DIVERSE_PROPOSAL"
        if ready else "KILL_EXACT_DEPTH3_PROGRAM",
        {
            "checks": checks,
            "observed_roots": observed,
            "observed_total": total,
            "maximum_family_share": max_share,
            "supported_alternative_families": supported_alternatives,
        },
    )


def _alternative_evidence() -> dict[str, Any]:
    g1r = _load_json(G1R_SIGNATURE_LOCK)
    action_audit = g1r["action_distinctness_audit"]
    qd = _load_json(QD_ADMISSION)
    qd5 = _load_json(G1R_QD5_SEAL)
    genuine = set(action_audit["representative_families"])
    if qd["decision"] == "READY_QD_FAMILY_ADMISSION":
        genuine.add("g1r_qd_static_archive_oneply_v2_terminal_schema")
    alternatives = {
        "g1r_expectimax2": 0,
        "g1r_qd_static_archive_oneply_v2_terminal_schema": 0,
    }
    return {
        "genuine_signature_family_count": len(genuine),
        "genuine_signature_families": sorted(genuine),
        "g1r_action_audit_sha256":
            canonical_json_hash(action_audit),
        "g1r_pairwise": action_audit["pairwise"],
        "qd_decision": qd["decision"],
        "qd_pairwise": qd["pairwise"],
        "qd5_seal_decision": qd5.get("decision"),
        "k1_compatible_observed_trigger_roots": alternatives,
        "support_interpretation":
            "distinct signatures exist; no immutable alternative-family "
            "metadata binds a natural root to the unchanged K1 trigger",
        "source_files": [
            {
                "path": str(path),
                "sha256": sha256_path(path),
            }
            for path in (G1R_SIGNATURE_LOCK, QD_ADMISSION, G1R_QD5_SEAL)
        ],
    }


def _unopened_audit() -> dict[str, Any]:
    c2_terminal = _load_json(C2_TERMINAL, require_payload_hash=True)
    checks = {
        "c2_terminal_exact":
            sha256_path(C2_TERMINAL) == EXPECTED_C2_TERMINAL_SHA256,
        "c2_killed_before_untouched_gate":
            c2_terminal["decision"] == "KILL_C2_COST_ADMISSION"
            and c2_terminal["stage"] == "engineering_validation",
        "c2_untouched_timing_absent":
            not (C2_TERMINAL.parent / "gate_timings.jsonl").exists()
            and not (
                C2_TERMINAL.parent / "C2_RUNTIME_GATE.json"
            ).exists(),
        "k1_terminal_exact":
            sha256_path(ROOT / "K1_TERMINAL_RESULT.json")
            == EXPECTED_K1_TERMINAL_SHA256,
        "k1_corpus_absent":
            not (ROOT / "K1_CORPUS_MANIFEST.json").exists(),
        "k1_fresh_gate_absent":
            not (ROOT / "K1_FRESH_ENGINEERING_GATE.json").exists(),
    }
    return {
        "checks": checks,
        "passes": all(checks.values()),
        "c2_terminal_sha256": sha256_path(C2_TERMINAL),
        "k1_terminal_sha256":
            sha256_path(ROOT / "K1_TERMINAL_RESULT.json"),
    }


def _operational_audit() -> dict[str, Any]:
    free = shutil.disk_usage(Path(".")).free / 1024**3
    services = k1.history.service_health()
    heavy = _heavy_process_audit()
    checks = {
        "free_disk_above_100_gib": free >= 100.0,
        "free_disk_above_120_gib_target": free >= 120.0,
        "services_healthy": bool(services["passes"]),
        "dashboard_top_three_exact":
            tuple(services.get("dashboard_top_scores", ()))
            == EXPECTED_TOP_THREE,
        "no_heavy_process": bool(heavy["passes"]),
    }
    return {
        "free_gib": free,
        "services": services,
        "heavy_process": heavy,
        "checks": checks,
        "passes": all(checks.values()),
    }


def run_audit(*, out_dir: Path) -> dict[str, Any]:
    if out_dir.resolve() != OUTPUT_DIR.resolve():
        raise ValueError("K1 support-audit output path mismatch")
    if out_dir.exists():
        raise FileExistsError(out_dir)
    if sha256_path(ROOT / "K1_TERMINAL_RESULT.json") != (
        EXPECTED_K1_TERMINAL_SHA256
    ):
        raise ValueError("K1 terminal changed")
    rows = _completion_rows()
    stream = _load_json(ROOT / "K1_STREAM_MANIFEST.json", require_payload_hash=True)
    stream_by_key = {
        (str(row["behavior_family"]), int(row["game_index"])): row
        for row in stream["rows"]
    }
    prior = _load_json(
        ROOT / "K1_EXCLUSION_MANIFEST.json",
        require_payload_hash=True,
    )
    prior_roots = set(str(root) for root in prior["root_tokens"])
    incumbent = make_policy(k1.incumbent_spec())
    if not isinstance(incumbent, NtupleExpectimaxPolicy):
        raise TypeError("K1 incumbent type mismatch")
    retained = []
    source_rows = []
    seen_roots: set[str] = set()
    all_stream_values = []
    integrity_checks = {
        "exact_108_rows": len(rows) == 108,
        "all_complete": all(bool(row["complete"]) for row in rows),
        "all_score_inspection_flags_false":
            all(row.get("score_inspected") is False for row in rows),
        "all_dashboard_ineligible":
            all(row.get("dashboard_eligible") is False for row in rows),
    }
    for completion in rows:
        key = (
            str(completion["behavior_family"]),
            int(completion["game_index"]),
        )
        expected_stream = stream_by_key[key]
        for name in k1.STREAM_BASES:
            value = int(completion[name])
            all_stream_values.append(value)
            if value != int(expected_stream[name]):
                raise ValueError(f"K1 stream mismatch: {key}: {name}")
        root = str(completion["root_ancestry"])
        if root in seen_roots or root in prior_roots:
            raise ValueError(f"K1 root integrity failure: {root}")
        seen_roots.add(root)
        replay_value = completion.get("source_replay")
        if replay_value is None:
            if completion.get("qualifying_root"):
                raise ValueError("Qualifying K1 root has no retained replay")
            continue
        replay_path = Path(str(replay_value))
        state_path = Path(str(completion["selected_states"]))
        if (
            sha256_path(replay_path) != completion["source_replay_sha256"]
            or sha256_path(state_path) != completion["selected_states_sha256"]
        ):
            raise ValueError("K1 retained-source hash mismatch")
        replay = _load_json(replay_path)
        selected = _load_json(state_path, require_payload_hash=True)
        reset = initial_reset_diagnostics(replay)
        if (
            not reset["is_reset_start"]
            or int(replay["seed"]) != int(completion["logical_seed"])
            or int(replay["starter_tile"]) != k1.STARTER_TILE
            or selected["root_ancestry"] != root
            or selected["source_replay_sha256"]
                != completion["source_replay_sha256"]
        ):
            raise ValueError("K1 retained-source provenance mismatch")
        qualifying = qualifying_states(
            replay,
            root=root,
            incumbent=incumbent,
        )
        if len(qualifying) < 4:
            raise ValueError("Retained K1 root no longer passes trigger screen")
        earliest = qualifying[0]
        selected_one = min(
            qualifying,
            key=lambda row: (
                _candidate_key(root, row),
                int(row["frame_index"]),
            ),
        )
        retained.append({
            "root_ancestry": root,
            "behavior_family": str(completion["behavior_family"]),
            "qualifying_state_count": len(qualifying),
            "earliest_qualifying_frame": earliest,
            "deterministic_one_state_selection": {
                **selected_one,
                "selection_sha256": _candidate_key(root, selected_one),
            },
        })
        source_rows.extend([
            {
                "path": str(replay_path),
                "sha256": completion["source_replay_sha256"],
            },
            {
                "path": str(state_path),
                "sha256": completion["selected_states_sha256"],
            },
        ])
    integrity_checks.update({
        "exact_108_unique_roots": len(seen_roots) == 108,
        "exact_432_unique_stream_ids":
            len(all_stream_values) == len(set(all_stream_values)) == 432,
        "zero_prior_root_overlap": not seen_roots.intersection(prior_roots),
        "retained_sources_hash_exact": len(source_rows) == 2 * len(retained),
    })
    family_support = summarize_family_support(
        completion_rows=rows,
        retained_rows=retained,
    )
    alternative = _alternative_evidence()
    unopened = _unopened_audit()
    operations = _operational_audit()
    integrity_passes = (
        all(integrity_checks.values())
        and unopened["passes"]
        and operations["passes"]
    )
    decision, decision_detail = root_diverse_decision(
        family_support,
        signature_families=alternative["genuine_signature_family_count"],
        integrity_passes=integrity_passes,
        alternative_trigger_support=
            alternative["k1_compatible_observed_trigger_roots"],
    )
    proposal = None
    if decision == "READY_K2_ROOT_DIVERSE_PROPOSAL":
        proposal = {
            "version": "k2_root_diverse_proposal_v1",
            "k1_v1_rerun": False,
            "target_roots_per_family": PREFERRED_ROOTS_PER_FAMILY,
            "minimum_roots_per_family": MIN_ROOTS_PER_FAMILY,
            "states_per_root": 1,
            "runtime_and_equivalence_gates": "unchanged_from_K1",
            "recommended_stream_namespace":
                "77B-80B subject to a future complete collision audit",
            "execution_authorized": False,
        }
    source_manifest = {
        "primary_files": [
            {
                "path": str(path),
                "sha256": sha256_path(path),
            }
            for path in (
                ROOT / "K1_PREFLIGHT_LOCK.json",
                ROOT / "K1_EXECUTION_OPENED.json",
                ROOT / "K1_TERMINAL_RESULT.json",
                ROOT / "K1_STREAM_MANIFEST.json",
                ROOT / "K1_POLICY_LOCK.json",
                ROOT / "K1_EXCLUSION_MANIFEST.json",
                ROOT / "completed_games.jsonl",
            )
        ],
        "retained_source_files": source_rows,
    }
    source_manifest["manifest_sha256"] = canonical_json_hash(source_manifest)
    result = payload_with_hash({
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "decision": decision,
        "charter": {
            "path": str(CHARTER_PATH),
            "sha256": sha256_path(CHARTER_PATH),
        },
        "implementation": {
            "path": str(RUNNER_PATH),
            "sha256": sha256_path(RUNNER_PATH),
        },
        "tests": {
            "path": str(TEST_PATH),
            "sha256": sha256_path(TEST_PATH),
        },
        "source_manifest": source_manifest,
        "integrity_checks": integrity_checks,
        "family_support": family_support,
        "retained_root_details": retained,
        "root_diverse_decision": decision_detail,
        "alternative_family_evidence": alternative,
        "unopened_gate_audit": unopened,
        "operations": operations,
        "k2_proposal": proposal,
        "interpretation":
            "feasibility only; missing replay content is unknown and cannot "
            "support a one-state-per-root design",
        "forbidden_work": {
            "new_games": 0,
            "streams_consumed": 0,
            "compilations": 0,
            "timings": 0,
            "depth3_values": 0,
            "policy_outcomes": 0,
            "scores_inspected": 0,
            "recorded_actions_inspected": 0,
            "labels": 0,
            "models": 0,
            "incumbent_changes": 0,
            "dashboard_changes": 0,
            "human_actions_used": 0,
        },
        "state": {
            "CONTINUE":
                "K2_proposal_only" if decision
                == "READY_K2_ROOT_DIVERSE_PROPOSAL" else "none",
            "HOLD": "compilation_timing_acquisition_policy_evaluation",
            "KILL": {
                "K1_v1": "spent_no_rerun",
                "exact_depth3_program":
                    decision == "KILL_EXACT_DEPTH3_PROGRAM",
            },
            "PROMOTE": False,
        },
    })
    out_dir.mkdir(parents=True)
    atomic_write_json(out_dir / "K1_SUPPORT_AUDIT.json", result)
    return {
        "decision": decision,
        "result_path": str(out_dir / "K1_SUPPORT_AUDIT.json"),
        "result_file_sha256":
            sha256_path(out_dir / "K1_SUPPORT_AUDIT.json"),
        "result_payload_sha256": result["canonical_payload_sha256"],
        "family_support": family_support,
        "root_diverse_decision": decision_detail,
        "unopened_gate_audit": unopened,
        "operations": operations,
        "forbidden_work": result["forbidden_work"],
        "state": result["state"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_audit(out_dir=args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

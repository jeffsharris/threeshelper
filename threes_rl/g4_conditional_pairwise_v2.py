"""G4-v2 ancestry-disjoint cross-fit on spent ordinary pair labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq

from threes_rl import g4_conditional_pairwise as v1
from threes_rl.g3_e0_label_fit import (
    canonical_sha256,
    json_object,
    verify_payload_hash,
    write_immutable_json,
)
from threes_rl.s3_power_preflight import sha256_path


VERSION = "g4_conditional_pairwise_v2"
AMENDMENT_PATH = Path("threes_rl/G4_V2_SPENT_CROSSFIT_AMENDMENT.md")
AMENDMENT_SHA256 = (
    "080d7bf3329986850bc9cb2d43409cb493ae11267513919b25e538c4a1296bca"
)
TEST_PATH = Path("tests/test_rl_g4_conditional_pairwise_v2.py")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/g4_conditional_pairwise_v2_test_evidence.json"
)
OUTPUT_DIR = Path("threes_rl/runs/forensics/g4_conditional_pairwise_v2")
FOLD_MANIFEST_NAME = "G4_V2_FOLD_MANIFEST.json"
PREFIT_LOCK_NAME = "G4_V2_PREFIT_LOCK.json"
OPEN_MARKER_NAME = "G4_V2_EXECUTION_OPENED.json"
PREDICTION_NAME = "G4_V2_OOF_PREDICTIONS.json"
TERMINAL_NAME = "G4_V2_TERMINAL_RESULT.json"
MODELS_DIR_NAME = "fold_models"

V1_PREFLIGHT_PATH = v1.OUTPUT_DIR / v1.PREFLIGHT_NAME
V1_PAIR_MANIFEST_PATH = v1.OUTPUT_DIR / v1.PAIR_MANIFEST_NAME
V1_ARTIFACT_HASHES = {
    str(v1.CHARTER_PATH):
        "765992cc0af3fc7c9d10c88ed3e0436a2ec6bc3b989f776775fe86230b22247e",
    "threes_rl/g4_conditional_pairwise.py":
        "d7fef45bb9d976b7912f6e12cde052fbd81c73589a3eed5029e8e9a1b95d2c27",
    str(v1.TEST_PATH):
        "67d2b91a07946788547ad2b860429ed7569c625f99ebf10d66ae0f3642fd0416",
    str(V1_PREFLIGHT_PATH):
        "bad6ca9542990144ae4d6872ef16781ec741bdb4c0584b5cf24e9783797155db",
    str(V1_PAIR_MANIFEST_PATH):
        "5acad327380b8cdc021a3299e085a06de656a0cacab4ba9984c59079db63602a",
    str(v1.ORDINARY_DB_PATH):
        "d0954a91e84bc7a420d64e7294f40232c1ffcb692fab86d07425b138e063f820",
    str(v1.G2_ROOT_MANIFEST_PATH):
        "60d514ed79ff315f7c2e0d2ad13bb712a57d4c3b204587691aa878a7486ea2ca",
    str(v1.G3_TERMINAL_PATH):
        "e7ca390f0c32ebb3a680235de02e12beb62f45b1050115e8c9a30a7a3ca0ddd1",
}
V1_PAIR_DATASET_SHA256 = (
    "ade1040d0f1bc56f58dfd0dc73004fa12f02d3caaae01880b08cf424d824484d"
)
FOLD_COUNT = 5
FOLD_HASH_NAMESPACE = "G4-v2-fivefold-v1"
BOOTSTRAP_SEED = 2_026_072_605
BOOTSTRAP_REPEATS = 10_000
MIN_OVERALL_ROOTS = 128
MIN_SCALE_ROOTS = 32
MIN_ELIGIBLE_FAMILIES = 3
MIN_FAMILY_ROOTS = 8
MAX_RAW_ROOT_SHARE = 0.10
POWER_ALPHA = 0.05
POWER_TARGET = 0.80


def _source_audit() -> dict[str, Any]:
    expected = {
        **V1_ARTIFACT_HASHES,
        str(AMENDMENT_PATH): AMENDMENT_SHA256,
    }
    actual = {
        path_text: (
            sha256_path(Path(path_text))
            if Path(path_text).is_file()
            else None
        )
        for path_text in expected
    }
    checks = {
        path_text: actual[path_text] == expected_hash
        for path_text, expected_hash in expected.items()
    }
    v1_preflight = json_object(V1_PREFLIGHT_PATH)
    v1_pair_manifest = json_object(V1_PAIR_MANIFEST_PATH)
    checks.update(
        {
            "v1_preflight_payload_exact":
                verify_payload_hash(v1_preflight)
                and v1_preflight.get("canonical_payload_sha256")
                == "4c0bc125ff2e14094d0bc6d330f3796458572af5bcf0ac22e53f0c6a6822a40e",
            "v1_decision_permanent":
                v1_preflight.get("decision") == "KILL_G4_PAIRWISE_INFEASIBLE",
            "v1_pair_manifest_payload_exact":
                verify_payload_hash(v1_pair_manifest),
            "v1_pair_dataset_exact":
                v1_preflight.get("pair_dataset_sha256")
                == V1_PAIR_DATASET_SHA256
                and v1_pair_manifest["dataset_audit"]["pair_audit"][
                    "pair_dataset_sha256"
                ]
                == V1_PAIR_DATASET_SHA256,
            "v1_diagnostic_never_opened":
                not (v1.OUTPUT_DIR / v1.DIAGNOSTIC_OPENED_NAME).exists()
                and not (v1.OUTPUT_DIR / v1.DIAGNOSTIC_RESULT_NAME).exists()
                and not (v1.OUTPUT_DIR / v1.MODEL_DIR_NAME).exists(),
            "g3_transfer_database_absent":
                not v1.FORBIDDEN_TRANSFER_DB_PATH.exists(),
            "g3_transfer_prediction_seal_absent":
                not v1.FORBIDDEN_TRANSFER_PREDICTION_PATH.exists(),
        }
    )
    return {
        "expected": expected,
        "actual": actual,
        "checks": checks,
        "passes": all(checks.values()),
        "g3_transfer_access": 0,
    }


def _load_test_evidence() -> dict[str, Any]:
    evidence = json_object(TEST_EVIDENCE_PATH)
    checks = {
        "payload": verify_payload_hash(evidence),
        "version": evidence.get("version") == f"{VERSION}_test_evidence",
        "amendment":
            evidence.get("amendment_sha256") == AMENDMENT_SHA256,
        "implementation":
            evidence.get("implementation_sha256")
            == sha256_path(Path(__file__)),
        "test":
            evidence.get("test_sha256") == sha256_path(TEST_PATH),
        "py_compile": bool(evidence.get("py_compile_passed")),
        "focused": int(evidence.get("focused_failed", -1)) == 0,
        "regressions":
            int(evidence.get("applicable_regressions_failed", -1)) == 0,
    }
    if not all(checks.values()):
        raise ValueError(
            "G4-v2 test evidence failed: "
            + ",".join(name for name, value in checks.items() if not value)
        )
    return evidence


def _ordinary_root_metadata() -> list[dict[str, Any]]:
    manifest = json_object(v1.G2_ROOT_MANIFEST_PATH)
    if not verify_payload_hash(manifest):
        raise ValueError("G2 root manifest payload hash mismatch")
    by_root: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for raw in manifest["records"]:
        if raw.get("partition") in v1.ACCEPTED_PARTITIONS:
            by_root[str(raw["root_cluster"])].append(raw)
    if sum(len(rows) for rows in by_root.values()) != v1.EXPECTED_ORDINARY_RECORDS:
        raise ValueError("Ordinary record metadata count changed")
    if len(by_root) != 352:
        raise ValueError(f"Expected 352 ordinary roots, found {len(by_root)}")

    roots = []
    for root, rows in sorted(by_root.items()):
        families = {str(row["behavior_family"]) for row in rows}
        partitions = {str(row["partition"]) for row in rows}
        scales = sorted({str(row["scale"]) for row in rows})
        if len(families) != 1 or len(partitions) != 1:
            raise ValueError(f"Root metadata fragmentation: {root}")
        if not set(scales).issubset(set(v1.ACCEPTED_SCALES)):
            raise ValueError(f"Unexpected ordinary scale: {root}")
        roots.append(
            {
                "root_cluster": root,
                "behavior_family": next(iter(families)),
                "partition": next(iter(partitions)),
                "scale_signature": "+".join(scales),
                "record_count": len(rows),
            }
        )
    return roots


def _fold_key(
    *,
    family: str,
    scale_signature: str,
    root: str,
) -> str:
    return hashlib.sha256(
        "|".join(
            (FOLD_HASH_NAMESPACE, family, scale_signature, root)
        ).encode("utf-8")
    ).hexdigest()


def build_fold_assignments(
    roots: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_stratum: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in roots:
        by_stratum[
            (str(row["behavior_family"]), str(row["scale_signature"]))
        ].append(row)
    assignments = []
    stratum_balance = []
    for (family, scale_signature), rows in sorted(by_stratum.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                _fold_key(
                    family=family,
                    scale_signature=scale_signature,
                    root=str(row["root_cluster"]),
                ),
                str(row["root_cluster"]),
            ),
        )
        counts = Counter()
        for index, row in enumerate(ordered):
            fold = index % FOLD_COUNT
            counts[fold] += 1
            assignments.append(
                {
                    "root_cluster": str(row["root_cluster"]),
                    "behavior_family": family,
                    "partition": str(row["partition"]),
                    "scale_signature": scale_signature,
                    "record_count": int(row["record_count"]),
                    "fold": fold,
                    "fold_key": _fold_key(
                        family=family,
                        scale_signature=scale_signature,
                        root=str(row["root_cluster"]),
                    ),
                }
            )
        stratum_balance.append(
            {
                "behavior_family": family,
                "scale_signature": scale_signature,
                "roots": len(ordered),
                "fold_counts": {
                    str(fold): counts.get(fold, 0)
                    for fold in range(FOLD_COUNT)
                },
                "max_minus_min": (
                    max(counts.get(fold, 0) for fold in range(FOLD_COUNT))
                    - min(counts.get(fold, 0) for fold in range(FOLD_COUNT))
                ),
            }
        )
    assignments.sort(key=lambda row: str(row["root_cluster"]))
    root_names = [str(row["root_cluster"]) for row in assignments]
    checks = {
        "root_count_352": len(assignments) == 352,
        "roots_unique": len(root_names) == len(set(root_names)),
        "folds_in_range": all(
            0 <= int(row["fold"]) < FOLD_COUNT for row in assignments
        ),
        "stratum_balance_at_most_one": all(
            int(row["max_minus_min"]) <= 1 for row in stratum_balance
        ),
        "partition_per_root": all(
            row["partition"] in v1.ACCEPTED_PARTITIONS for row in assignments
        ),
    }
    return assignments, {
        "fold_count": FOLD_COUNT,
        "assignment_count": len(assignments),
        "assignment_sha256": canonical_sha256(assignments),
        "fold_root_counts": {
            str(fold): sum(
                int(row["fold"]) == fold for row in assignments
            )
            for fold in range(FOLD_COUNT)
        },
        "stratum_balance": stratum_balance,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _minimum_detectable_rate(n: int) -> float:
    if n <= 0:
        raise ValueError("Power root count must be positive")
    return float(
        brentq(
            lambda probability: (
                v1._exact_binomial_power(n, probability, alpha=POWER_ALPHA)
                - POWER_TARGET
            ),
            0.500001,
            0.999999,
        )
    )


def _v1_support_summary() -> dict[str, Any]:
    preflight = json_object(V1_PREFLIGHT_PATH)
    train = dict(preflight["train_support"])
    development = dict(preflight["development_support"])
    if (
        int(train["pairs"]) != 552
        or int(development["pairs"]) != 175
        or int(train["roots"]) != 146
        or int(development["roots"]) != 39
    ):
        raise ValueError("V1 aggregate support identity changed")
    family_counts = Counter(
        {
            str(family): int(count)
            for family, count in train["roots_by_family"].items()
        }
    )
    family_counts.update(
        {
            str(family): int(count)
            for family, count in development["roots_by_family"].items()
        }
    )
    scale_counts = {
        scale: (
            int(train["roots_by_scale"].get(scale, 0))
            + int(development["roots_by_scale"].get(scale, 0))
        )
        for scale in v1.ACCEPTED_SCALES
    }
    max_train_pairs = int(
        round(
            float(train["max_raw_pair_share_by_root"])
            * int(train["pairs"])
        )
    )
    max_development_pairs = int(
        round(
            float(development["max_raw_pair_share_by_root"])
            * int(development["pairs"])
        )
    )
    total_roots = int(train["roots"]) + int(development["roots"])
    total_pairs = int(train["pairs"]) + int(development["pairs"])
    eligible_families = {
        family: count
        for family, count in sorted(family_counts.items())
        if count >= MIN_FAMILY_ROOTS
    }
    power = {
        "overall": {
            "roots": total_roots,
            "mde_true_root_direction_rate": _minimum_detectable_rate(
                total_roots
            ),
        },
        "by_scale": {
            scale: {
                "roots": count,
                "mde_true_root_direction_rate": _minimum_detectable_rate(
                    count
                ),
            }
            for scale, count in scale_counts.items()
        },
        "by_eligible_family": {
            family: {
                "roots": count,
                "mde_true_root_direction_rate": _minimum_detectable_rate(
                    count
                ),
            }
            for family, count in eligible_families.items()
        },
        "test": "exact two-sided binomial against root direction 0.50",
        "alpha": POWER_ALPHA,
        "target_power": POWER_TARGET,
    }
    return {
        "pair_dataset_sha256": preflight["pair_dataset_sha256"],
        "train": train,
        "development": development,
        "total_pairs": total_pairs,
        "total_informative_roots": total_roots,
        "informative_roots_by_scale": scale_counts,
        "informative_roots_by_family": dict(sorted(family_counts.items())),
        "support_eligible_families": eligible_families,
        "conservative_max_pairs_one_root": max(
            max_train_pairs, max_development_pairs
        ),
        "conservative_max_raw_root_share": (
            max(max_train_pairs, max_development_pairs) / total_pairs
        ),
        "power": power,
    }


def _prefit_support_checks(
    support: Mapping[str, Any],
    *,
    root_metadata_audit: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "informative_roots_at_least_128":
            int(support["total_informative_roots"]) >= MIN_OVERALL_ROOTS,
        "each_scale_at_least_32": all(
            int(value) >= MIN_SCALE_ROOTS
            for value in support["informative_roots_by_scale"].values()
        ),
        "three_families_at_least_8":
            len(support["support_eligible_families"])
            >= MIN_ELIGIBLE_FAMILIES,
        "max_raw_root_share_at_most_0_10":
            float(support["conservative_max_raw_root_share"])
            <= MAX_RAW_ROOT_SHARE,
        "ordinary_metadata_root_disjoint":
            bool(root_metadata_audit["passes"]),
    }


def run_prefit(out_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"G4-v2 output already exists: {out_dir}")
    source_audit = _source_audit()
    if not source_audit["passes"]:
        raise ValueError("G4-v2 immutable source audit failed")
    roots = _ordinary_root_metadata()
    assignments, fold_audit = build_fold_assignments(roots)
    support = _v1_support_summary()
    test_evidence = _load_test_evidence()
    support_checks = _prefit_support_checks(
        support,
        root_metadata_audit=fold_audit,
    )
    operations = v1._operational_audit()
    integrity_checks = {
        "source_audit": bool(source_audit["passes"]),
        "fold_assignment": bool(fold_audit["passes"]),
        "v1_pair_identity":
            support["pair_dataset_sha256"] == V1_PAIR_DATASET_SHA256,
        "g3_transfer_access_zero": True,
        "operations": bool(operations["passes"]),
    }
    ready = all(support_checks.values()) and all(integrity_checks.values())
    decision = (
        "READY_G4_V2_CROSSFIT_EXECUTION"
        if ready
        else "HOLD_G4_V2_CROSSFIT_UNDERPOWERED"
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    fold_manifest = {
        "version": f"{VERSION}_fold_manifest",
        "amendment_sha256": AMENDMENT_SHA256,
        "outcomes_opened": False,
        "assignment_method": {
            "folds": FOLD_COUNT,
            "stratum": "(behavior_family,sorted_scale_signature)",
            "order": (
                "SHA256(G4-v2-fivefold-v1|family|scale_signature|root),"
                "then literal root"
            ),
            "assignment": "ordered_index_mod_5",
        },
        "assignments": assignments,
        "audit": fold_audit,
    }
    fold_manifest["canonical_payload_sha256"] = canonical_sha256(
        fold_manifest
    )
    write_immutable_json(out_dir / FOLD_MANIFEST_NAME, fold_manifest)

    implementation_path = Path(__file__)
    execute_command = (
        "zsh -ic 'no-secrets nice -n 10 env PYTHONPATH=. "
        ".venv/bin/python -m threes_rl.g4_conditional_pairwise_v2 execute "
        f"--out-dir {out_dir} --prefit-lock {out_dir / PREFIT_LOCK_NAME}'"
    )
    lock = {
        "version": f"{VERSION}_prefit_lock",
        "decision": decision,
        "amendment_path": str(AMENDMENT_PATH),
        "amendment_sha256": AMENDMENT_SHA256,
        "implementation_path": str(implementation_path),
        "implementation_sha256": sha256_path(implementation_path),
        "test_path": str(TEST_PATH),
        "test_sha256": sha256_path(TEST_PATH),
        "test_evidence_path": str(TEST_EVIDENCE_PATH),
        "test_evidence_file_sha256": sha256_path(TEST_EVIDENCE_PATH),
        "test_evidence_payload_sha256":
            test_evidence["canonical_payload_sha256"],
        "v1_source_audit": source_audit,
        "pair_dataset_sha256": V1_PAIR_DATASET_SHA256,
        "support": support,
        "support_checks": support_checks,
        "fold_manifest_path": str(out_dir / FOLD_MANIFEST_NAME),
        "fold_manifest_file_sha256": sha256_path(
            out_dir / FOLD_MANIFEST_NAME
        ),
        "fold_manifest_payload_sha256":
            fold_manifest["canonical_payload_sha256"],
        "integrity_checks": integrity_checks,
        "operations": operations,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repeats": BOOTSTRAP_REPEATS,
            "unit": "whole ancestry root direction",
            "lower_quantile": 0.025,
            "upper_quantile": 0.975,
        },
        "material_overall_point_floor":
            support["power"]["overall"]["mde_true_root_direction_rate"],
        "model_contract": {
            "feature_width": v1.FEATURE_WIDTH,
            "intercept": False,
            "l2_lambda": v1.L2_LAMBDA,
            "optimizer": "L-BFGS-B",
            "maxiter": v1.MAX_OPTIMIZER_ITERATIONS,
            "gtol": v1.OPTIMIZER_GTOL,
            "calibration": None,
            "folds": FOLD_COUNT,
            "training_weights":
                "v1 family/root/record/unit/pair balanced",
            "evaluation_weights":
                "root/record/unit/pair balanced natural root mixture",
        },
        "execute_command": execute_command,
        "test_evidence": test_evidence,
        "g3_transfer_access": {
            "records": 0,
            "predictions": 0,
            "paths": 0,
            "database_opened": False,
        },
        "forbidden_work": {
            "pair_outcomes_reopened": 0,
            "model_fits": 0,
            "new_labels": 0,
            "simulations": 0,
            "policy_outcomes": 0,
            "scores_inspected": 0,
            "dashboard_changes": 0,
        },
        "state": {
            "CONTINUE": (
                "one_spent_crossfit_execution" if ready else "none"
            ),
            "HOLD":
                "fresh_acquisition_labels_policy_work_C2_human_PROMOTE",
            "KILL": "G3_and_G4_v1_permanent",
            "PROMOTE": False,
        },
    }
    lock["canonical_payload_sha256"] = canonical_sha256(lock)
    write_immutable_json(out_dir / PREFIT_LOCK_NAME, lock)
    return lock


def _validate_prefit_lock(
    out_dir: Path,
    prefit_lock: Path,
    *,
    require_ready: bool = True,
) -> dict[str, Any]:
    if prefit_lock.resolve() != (out_dir / PREFIT_LOCK_NAME).resolve():
        raise ValueError("G4-v2 prefit lock path mismatch")
    lock = json_object(prefit_lock)
    checks = {
        "payload": verify_payload_hash(lock),
        "version": lock.get("version") == f"{VERSION}_prefit_lock",
        "decision": (
            lock.get("decision") == "READY_G4_V2_CROSSFIT_EXECUTION"
            if require_ready else True
        ),
        "amendment": (
            lock.get("amendment_sha256") == AMENDMENT_SHA256
            and sha256_path(AMENDMENT_PATH) == AMENDMENT_SHA256
        ),
        "implementation":
            lock.get("implementation_sha256") == sha256_path(Path(__file__)),
        "test": lock.get("test_sha256") == sha256_path(TEST_PATH),
        "test_evidence": (
            lock.get("test_evidence_file_sha256")
            == sha256_path(TEST_EVIDENCE_PATH)
            and lock.get("test_evidence_payload_sha256")
            == _load_test_evidence()["canonical_payload_sha256"]
        ),
        "fold_file":
            lock.get("fold_manifest_file_sha256")
            == sha256_path(out_dir / FOLD_MANIFEST_NAME),
        "source": bool(_source_audit()["passes"]),
        "pair_identity":
            lock.get("pair_dataset_sha256") == V1_PAIR_DATASET_SHA256,
    }
    if not all(checks.values()):
        raise ValueError(
            "G4-v2 prefit lock validation failed: "
            + ",".join(name for name, value in checks.items() if not value)
        )
    return lock


def open_execution(
    out_dir: Path = OUTPUT_DIR,
    prefit_lock: Path | None = None,
) -> dict[str, Any]:
    lock_path = prefit_lock or out_dir / PREFIT_LOCK_NAME
    lock = _validate_prefit_lock(out_dir, lock_path)
    marker_path = out_dir / OPEN_MARKER_NAME
    terminal_path = out_dir / TERMINAL_NAME
    if (
        marker_path.exists()
        or terminal_path.exists()
        or (out_dir / PREDICTION_NAME).exists()
        or (out_dir / MODELS_DIR_NAME).exists()
    ):
        raise FileExistsError("G4-v2 execution is one-shot and already opened")
    operations = v1._operational_audit()
    if not operations["passes"]:
        raise ValueError("G4-v2 open operational audit failed")
    marker = {
        "version": f"{VERSION}_execution_opened",
        "amendment_sha256": AMENDMENT_SHA256,
        "prefit_lock_path": str(lock_path),
        "prefit_lock_file_sha256": sha256_path(lock_path),
        "prefit_lock_payload_sha256": lock["canonical_payload_sha256"],
        "fold_manifest_file_sha256":
            lock["fold_manifest_file_sha256"],
        "pair_dataset_sha256": V1_PAIR_DATASET_SHA256,
        "implementation_sha256": sha256_path(Path(__file__)),
        "test_sha256": sha256_path(TEST_PATH),
        "execute_command": lock["execute_command"],
        "operations": operations,
        "zero_work_before_marker": {
            "pair_outcomes_reopened": 0,
            "models": 0,
            "predictions": 0,
            "new_labels": 0,
            "simulations": 0,
            "transfer_access": 0,
        },
    }
    marker["canonical_payload_sha256"] = canonical_sha256(marker)
    write_immutable_json(marker_path, marker)
    return marker


def _validate_marker(
    out_dir: Path,
    prefit_lock: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _validate_prefit_lock(out_dir, prefit_lock)
    marker = json_object(out_dir / OPEN_MARKER_NAME)
    checks = {
        "payload": verify_payload_hash(marker),
        "version": marker.get("version") == f"{VERSION}_execution_opened",
        "lock_file":
            marker.get("prefit_lock_file_sha256")
            == sha256_path(prefit_lock),
        "lock_payload":
            marker.get("prefit_lock_payload_sha256")
            == lock["canonical_payload_sha256"],
        "fold":
            marker.get("fold_manifest_file_sha256")
            == lock["fold_manifest_file_sha256"],
        "pair":
            marker.get("pair_dataset_sha256") == V1_PAIR_DATASET_SHA256,
        "implementation":
            marker.get("implementation_sha256") == sha256_path(Path(__file__)),
        "test": marker.get("test_sha256") == sha256_path(TEST_PATH),
        "command": marker.get("execute_command") == lock["execute_command"],
    }
    if not all(checks.values()):
        raise ValueError(
            "G4-v2 marker validation failed: "
            + ",".join(name for name, value in checks.items() if not value)
        )
    return lock, marker


def _root_local_weights(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    inherited = v1.assign_pair_weights(rows)
    roots = {str(row["root_cluster"]) for row in inherited}
    result = []
    for row in inherited:
        item = dict(row)
        item["evaluation_weight"] = (
            float(row["root_local_weight"]) / len(roots)
        )
        result.append(item)
    total = sum(float(row["evaluation_weight"]) for row in result)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise ValueError("Root-balanced evaluation weights do not sum to one")
    return result


def _score_heldout_rows(
    model: v1.PairwiseModel,
    rows: Sequence[Mapping[str, Any]],
    *,
    fold: int,
) -> list[dict[str, Any]]:
    matrix = np.stack(
        [np.asarray(row["delta"], dtype=np.float64) for row in rows]
    )
    logits = model.logits(matrix)
    reverse_logits = model.logits(-matrix)
    if not np.allclose(reverse_logits, -logits, rtol=0.0, atol=1e-12):
        raise ValueError("Pairwise antisymmetry failed")
    scored = []
    for row, logit in zip(rows, logits):
        label = int(row["label"])
        if float(logit) > 0.0:
            concordance = float(label)
        elif float(logit) < 0.0:
            concordance = float(1 - label)
        else:
            concordance = 0.5
        scored.append(
            {
                **dict(row),
                "fold": fold,
                "logit": float(logit),
                "concordance": concordance,
            }
        )
    return scored


def _root_values(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    weighted = _root_local_weights(rows)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in weighted:
        grouped[str(row["root_cluster"])].append(row)
    result = {}
    for root, local_rows in sorted(grouped.items()):
        local_total = sum(
            float(row["root_local_weight"]) for row in local_rows
        )
        continuous = sum(
            float(row["root_local_weight"]) * float(row["concordance"])
            for row in local_rows
        ) / local_total
        if continuous > 0.5:
            direction = 1.0
        elif continuous < 0.5:
            direction = 0.0
        else:
            direction = 0.5
        result[root] = {
            "root_cluster": root,
            "behavior_family": str(local_rows[0]["behavior_family"]),
            "continuous_concordance": float(continuous),
            "direction": direction,
            "pairs": len(local_rows),
        }
    return result


def root_balanced_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "pairs": 0,
            "roots": 0,
            "root_direction_rate": None,
            "continuous_root_concordance": None,
        }
    roots = _root_values(rows)
    values = list(roots.values())
    return {
        "pairs": len(rows),
        "roots": len(values),
        "root_direction_rate": float(
            np.mean([float(row["direction"]) for row in values])
        ),
        "continuous_root_concordance": float(
            np.mean(
                [float(row["continuous_concordance"]) for row in values]
            )
        ),
        "root_direction_ties": sum(
            float(row["direction"]) == 0.5 for row in values
        ),
        "pair_prediction_ties": sum(
            float(row["logit"]) == 0.0 for row in rows
        ),
        "max_raw_root_pair_share": (
            max(Counter(str(row["root_cluster"]) for row in rows).values())
            / len(rows)
        ),
    }


def root_direction_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    roots = _root_values(rows)
    values = np.asarray(
        [float(roots[root]["direction"]) for root in sorted(roots)],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("No roots for OOF bootstrap")
    rng = np.random.default_rng(seed)
    chunk = 1_000
    draws = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, chunk):
        stop = min(start + chunk, repeats)
        sample = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
        )
        draws[start:stop] = np.mean(values[sample], axis=1)
    return {
        "seed": seed,
        "repeats": repeats,
        "unit": "whole ancestry root direction",
        "point": float(np.mean(values)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def _subset_summaries(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    values = sorted({str(row[key]) for row in rows})
    return {
        value: root_balanced_summary(
            [row for row in rows if str(row[key]) == value]
        )
        for value in values
    }


def _terminal_decision(
    *,
    primary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    by_scale: Mapping[str, Mapping[str, Any]],
    by_family: Mapping[str, Mapping[str, Any]],
    eligible_families: Mapping[str, int],
    material_floor: float,
    model_checks: Mapping[str, bool],
) -> tuple[str, dict[str, bool]]:
    family_positive = sum(
        float(by_family[family]["root_direction_rate"]) > 0.5
        for family in eligible_families
    )
    checks = {
        "material_point_floor":
            float(primary["root_direction_rate"]) >= material_floor,
        "bootstrap_lower_above_half":
            float(bootstrap["lower_95"]) > 0.5,
        "both_scales_above_half": all(
            float(by_scale[scale]["root_direction_rate"]) > 0.5
            for scale in v1.ACCEPTED_SCALES
        ),
        "three_supported_families_above_half":
            family_positive >= MIN_ELIGIBLE_FAMILIES,
        "model_and_integrity": all(model_checks.values()),
    }
    decision = (
        "READY_G4_FRESH_ROOT_ACQUISITION_PREFLIGHT"
        if all(checks.values())
        else "KILL_G4_PAIRWISE_MECHANISM"
    )
    return decision, {
        **checks,
        "support_eligible_family_count":
            len(eligible_families),
        "support_eligible_families_positive": family_positive,
    }


def _prediction_payload_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "partition": str(row["partition"]),
            "scale": str(row["scale"]),
            "behavior_family": str(row["behavior_family"]),
            "root_cluster": str(row["root_cluster"]),
            "record_id": str(row["record_id"]),
            "horizon": str(row["horizon"]),
            "replicate": int(row["replicate"]),
            "action_pair": str(row["action_pair"]),
            "action_a_id": int(row["action_a_id"]),
            "action_b_id": int(row["action_b_id"]),
            "label": int(row["label"]),
            "fold": int(row["fold"]),
            "logit": float(row["logit"]),
            "concordance": float(row["concordance"]),
        }
        for row in rows
    ]


def execute_crossfit(
    out_dir: Path = OUTPUT_DIR,
    prefit_lock: Path | None = None,
) -> dict[str, Any]:
    lock_path = prefit_lock or out_dir / PREFIT_LOCK_NAME
    lock, marker = _validate_marker(out_dir, lock_path)
    terminal_path = out_dir / TERMINAL_NAME
    if terminal_path.exists():
        terminal = json_object(terminal_path)
        if not verify_payload_hash(terminal):
            raise ValueError("Existing G4-v2 terminal hash mismatch")
        return terminal

    try:
        pairs, dataset_audit, _sources = v1._build_dataset()
        pair_sha = dataset_audit["pair_audit"]["pair_dataset_sha256"]
        if pair_sha != V1_PAIR_DATASET_SHA256 or len(pairs) != 727:
            raise ValueError("G4-v2 pair reconstruction identity failed")
        fold_manifest = json_object(out_dir / FOLD_MANIFEST_NAME)
        if not verify_payload_hash(fold_manifest):
            raise ValueError("G4-v2 fold manifest payload mismatch")
        fold_by_root = {
            str(row["root_cluster"]): int(row["fold"])
            for row in fold_manifest["assignments"]
        }
        pair_roots = {str(row["root_cluster"]) for row in pairs}
        if not pair_roots.issubset(fold_by_root):
            raise ValueError("Informative root missing frozen fold")

        models_root = out_dir / MODELS_DIR_NAME
        models_root.mkdir(exist_ok=True)
        all_scored = []
        model_artifacts = []
        source_base = {
            "amendment": AMENDMENT_SHA256,
            "prefit_lock": sha256_path(lock_path),
            "fold_manifest": sha256_path(out_dir / FOLD_MANIFEST_NAME),
            "v1_pair_dataset": V1_PAIR_DATASET_SHA256,
            "implementation": sha256_path(Path(__file__)),
            "test": sha256_path(TEST_PATH),
        }
        fold_root_leakage = {}
        for fold in range(FOLD_COUNT):
            train_rows = [
                row
                for row in pairs
                if fold_by_root[str(row["root_cluster"])] != fold
            ]
            heldout_rows = [
                row
                for row in pairs
                if fold_by_root[str(row["root_cluster"])] == fold
            ]
            train_roots = {str(row["root_cluster"]) for row in train_rows}
            heldout_roots = {
                str(row["root_cluster"]) for row in heldout_rows
            }
            overlap = sorted(train_roots.intersection(heldout_roots))
            fold_root_leakage[str(fold)] = overlap
            if overlap or not heldout_rows or not train_rows:
                raise ValueError(f"Fold {fold} ancestry leakage/emptiness")
            model_sources = {
                **source_base,
                "heldout_fold": str(fold),
            }
            model_dir = models_root / f"fold_{fold}"
            if model_dir.exists():
                model = v1.PairwiseModel.load(
                    model_dir,
                    expected_source_hashes=model_sources,
                )
                if model.pair_dataset_sha256 != V1_PAIR_DATASET_SHA256:
                    raise ValueError("Resumed fold pair identity mismatch")
            else:
                model = v1.fit_pairwise_model(
                    train_rows,
                    source_hashes=model_sources,
                    pair_dataset_sha256=V1_PAIR_DATASET_SHA256,
                )
                model.save(model_dir)
            loaded = v1.PairwiseModel.load(
                model_dir,
                expected_source_hashes=model_sources,
            )
            scored = _score_heldout_rows(
                loaded,
                heldout_rows,
                fold=fold,
            )
            all_scored.extend(scored)
            model_artifacts.append(
                {
                    "fold": fold,
                    "train_pairs": len(train_rows),
                    "heldout_pairs": len(heldout_rows),
                    "train_roots": len(train_roots),
                    "heldout_roots": len(heldout_roots),
                    "meta_file_sha256": sha256_path(
                        model_dir / "meta.json"
                    ),
                    "arrays_file_sha256": sha256_path(
                        model_dir / "arrays.npz"
                    ),
                    "optimizer": loaded.optimizer_summary,
                    "coefficient_sha256": hashlib.sha256(
                        np.asarray(
                            loaded.coefficients,
                            dtype="<f8",
                        ).tobytes()
                    ).hexdigest(),
                    "save_load_prediction_exact": bool(
                        np.array_equal(
                            model.logits(
                                np.stack(
                                    [
                                        np.asarray(
                                            row["delta"],
                                            dtype=np.float64,
                                        )
                                        for row in heldout_rows
                                    ]
                                )
                            ),
                            loaded.logits(
                                np.stack(
                                    [
                                        np.asarray(
                                            row["delta"],
                                            dtype=np.float64,
                                        )
                                        for row in heldout_rows
                                    ]
                                )
                            ),
                        )
                    ),
                }
            )

        if len(all_scored) != len(pairs):
            raise ValueError("OOF prediction coverage mismatch")
        all_scored.sort(
            key=lambda row: (
                str(row["partition"]),
                str(row["behavior_family"]),
                str(row["root_cluster"]),
                str(row["record_id"]),
                v1.HORIZON_NAMES.index(str(row["horizon"])),
                int(row["replicate"]),
                int(row["action_a_id"]),
                int(row["action_b_id"]),
            )
        )
        prediction_rows = _prediction_payload_rows(all_scored)
        prediction_payload = {
            "version": f"{VERSION}_oof_predictions",
            "amendment_sha256": AMENDMENT_SHA256,
            "pair_dataset_sha256": V1_PAIR_DATASET_SHA256,
            "fold_manifest_file_sha256": sha256_path(
                out_dir / FOLD_MANIFEST_NAME
            ),
            "rows": prediction_rows,
            "rows_sha256": canonical_sha256(prediction_rows),
            "row_count": len(prediction_rows),
            "new_labels_generated": 0,
            "g3_transfer_access": 0,
        }
        prediction_payload["canonical_payload_sha256"] = canonical_sha256(
            prediction_payload
        )
        prediction_path = out_dir / PREDICTION_NAME
        if prediction_path.exists():
            existing = json_object(prediction_path)
            if existing != prediction_payload:
                raise ValueError("Existing OOF prediction payload mismatch")
        else:
            write_immutable_json(prediction_path, prediction_payload)

        primary = root_balanced_summary(all_scored)
        bootstrap = root_direction_bootstrap(all_scored)
        by_scale = _subset_summaries(all_scored, "scale")
        by_family = _subset_summaries(all_scored, "behavior_family")
        by_fold = {
            str(fold): root_balanced_summary(
                [row for row in all_scored if int(row["fold"]) == fold]
            )
            for fold in range(FOLD_COUNT)
        }
        by_horizon = _subset_summaries(all_scored, "horizon")
        by_action_pair = _subset_summaries(all_scored, "action_pair")
        model_checks = {
            "five_models": len(model_artifacts) == FOLD_COUNT,
            "all_optimizer_success": all(
                bool(row["optimizer"].get("success"))
                for row in model_artifacts
            ),
            "all_gradients_at_most_1e_4": all(
                float(
                    row["optimizer"].get(
                        "gradient_infinity_norm", math.inf
                    )
                )
                <= 1e-4
                for row in model_artifacts
            ),
            "all_save_load_exact": all(
                bool(row["save_load_prediction_exact"])
                for row in model_artifacts
            ),
            "zero_fold_root_leakage": not any(
                fold_root_leakage.values()
            ),
            "all_predictions_finite": all(
                math.isfinite(float(row["logit"])) for row in all_scored
            ),
            "all_727_pairs_predicted": len(all_scored) == 727,
            "all_185_roots_predicted": int(primary["roots"]) == 185,
            "transfer_access_zero": True,
        }
        operations = v1._operational_audit()
        model_checks["operational_health"] = bool(operations["passes"])
        decision, decision_checks = _terminal_decision(
            primary=primary,
            bootstrap=bootstrap,
            by_scale=by_scale,
            by_family=by_family,
            eligible_families=lock["support"][
                "support_eligible_families"
            ],
            material_floor=float(
                lock["material_overall_point_floor"]
            ),
            model_checks=model_checks,
        )
        terminal = {
            "version": f"{VERSION}_terminal",
            "decision": decision,
            "amendment_sha256": AMENDMENT_SHA256,
            "prefit_lock_file_sha256": sha256_path(lock_path),
            "prefit_lock_payload_sha256":
                lock["canonical_payload_sha256"],
            "marker_file_sha256": sha256_path(
                out_dir / OPEN_MARKER_NAME
            ),
            "marker_payload_sha256":
                marker["canonical_payload_sha256"],
            "pair_dataset_sha256": pair_sha,
            "fold_manifest_file_sha256": sha256_path(
                out_dir / FOLD_MANIFEST_NAME
            ),
            "prediction_file_sha256": sha256_path(prediction_path),
            "prediction_payload_sha256":
                prediction_payload["canonical_payload_sha256"],
            "model_artifacts": model_artifacts,
            "model_checks": model_checks,
            "decision_checks": decision_checks,
            "material_overall_point_floor":
                lock["material_overall_point_floor"],
            "primary_root_balanced": primary,
            "primary_bootstrap": bootstrap,
            "by_scale": by_scale,
            "by_family": by_family,
            "by_fold_descriptive": by_fold,
            "by_horizon": by_horizon,
            "by_action_pair": by_action_pair,
            "fold_root_leakage": fold_root_leakage,
            "operations": operations,
            "g3_transfer_access": {
                "records": 0,
                "predictions": 0,
                "paths": 0,
                "database_opened": False,
            },
            "forbidden_work": {
                "new_simulations": 0,
                "new_labels": 0,
                "policy_construction": 0,
                "policy_evaluation": 0,
                "scores_inspected": 0,
                "incumbent_changes": 0,
                "dashboard_changes": 0,
            },
            "state": {
                "CONTINUE": (
                    "await_research_lead_for_outcome_free_acquisition_preflight"
                    if decision
                    == "READY_G4_FRESH_ROOT_ACQUISITION_PREFLIGHT"
                    else "none"
                ),
                "HOLD":
                    "fresh_acquisition_labels_policy_work_C2_human_PROMOTE",
                "KILL": {
                    "G3": "permanent",
                    "G4_v1": "permanent",
                    "G4_pairwise_mechanism": (
                        decision == "KILL_G4_PAIRWISE_MECHANISM"
                    ),
                },
                "PROMOTE": False,
            },
        }
        terminal["canonical_payload_sha256"] = canonical_sha256(terminal)
        write_immutable_json(terminal_path, terminal)
        return terminal
    except Exception as error:
        if terminal_path.exists():
            raise
        terminal = {
            "version": f"{VERSION}_terminal",
            "decision": "HOLD_G4_V2_CROSSFIT_UNDERPOWERED",
            "failure_kind": "execution_integrity_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "amendment_sha256": AMENDMENT_SHA256,
            "prefit_lock_file_sha256": sha256_path(lock_path),
            "marker_file_sha256": sha256_path(
                out_dir / OPEN_MARKER_NAME
            ),
            "g3_transfer_access": 0,
            "new_labels_generated": 0,
            "policy_outcomes": 0,
            "PROMOTE": False,
        }
        terminal["canonical_payload_sha256"] = canonical_sha256(terminal)
        write_immutable_json(terminal_path, terminal)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G4-v2 spent ordinary conditional-pairwise cross-fit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prefit = subparsers.add_parser("prefit")
    prefit.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    for command in ("open", "execute"):
        child = subparsers.add_parser(command)
        child.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
        child.add_argument("--prefit-lock", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prefit":
        result = run_prefit(args.out_dir)
    elif args.command == "open":
        result = open_execution(args.out_dir, args.prefit_lock)
    else:
        result = execute_crossfit(args.out_dir, args.prefit_lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

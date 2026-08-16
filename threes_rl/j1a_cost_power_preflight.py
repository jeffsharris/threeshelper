"""Outcome-free J1a cost/power amendment and readiness tooling.

This module has no execution marker, stream reservation, game, training, or
outcome command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from threes_rl import j1_joint_policy_value as j1
from threes_rl.o2_online_option_preflight import simulate_capability_power


VERSION = "j1a_cost_power_amendment_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = (
    REPO_ROOT / "threes_rl" / "J1A_OUTCOME_FREE_COST_POWER_AMENDMENT.md"
)
RUNNER_PATH = REPO_ROOT / "threes_rl" / "j1a_cost_power_preflight.py"
TEST_PATH = REPO_ROOT / "tests" / "test_rl_j1a_cost_power_preflight.py"
OUTPUT_DIR = (
    REPO_ROOT
    / "threes_rl"
    / "runs"
    / "forensics"
    / "j1a_cost_power_amendment_v1"
)

TEST_EVIDENCE_NAME = "J1A_TEST_EVIDENCE.json"
ARITHMETIC_NAME = "J1A_COST_POWER_ARITHMETIC.json"
PREFLIGHT_LOCK_NAME = "J1A_PREFLIGHT_LOCK.json"
PREFLIGHT_RESULT_NAME = "J1A_PREFLIGHT_RESULT.json"

FOCUSED_TEST_COMMAND = (
    "env PYTHONPATH=. .venv/bin/python -m pytest -q "
    "tests/test_rl_j1a_cost_power_preflight.py"
)
PARENT_TEST_COMMAND = (
    "env PYTHONPATH=. .venv/bin/python -m pytest -q "
    "tests/test_rl_j1_joint_policy_value.py"
)
FOCUSED_TEST_COUNT = 18
PARENT_TEST_COUNT = 36

TRAIN_ROOTS = 16_384
DEVELOPMENT_PAIRS = 896
CONFIRMATION_PAIRS = 4_480
TOTAL_GAME_ARMS = (
    TRAIN_ROOTS + 2 * DEVELOPMENT_PAIRS + 2 * CONFIRMATION_PAIRS
)

PLANNING_MOVES = 512
MAX_MOVES = 5_000
SAFETY_MULTIPLIER = 1.25
RUNTIME_HEADROOM_MAX_FRACTION = 0.91
SCORE_SD = 1.25
SCORE_EFFECT = 0.07
SCORE_REQUIRED_POWER = 0.95
SCORE_MDE_MAX = 0.055
SCORE_Z_975 = 1.959963984540054
SCORE_Z_80 = 0.8416212335729143

CONTROL_RATES = (0.02, 0.04, 0.08, 0.15)
COUPLINGS = (0.0, 0.05, 0.10)
ODDS_RATIO_GRID = (1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
STREAM_STRATA = 8
POWER_DATASETS = 768
POWER_BOOTSTRAPS = 199
POWER_POINT_GATE = 1.25
POWER_LOWER_GATE = 1.0
POWER_REQUIRED = 0.80
POWER_CALIBRATION_NAME = "J1"

EXPECTED_O2_POWER_SOURCE_SHA256 = (
    "99e61f551d607e3b5b8457b7e76a17c8540f0e1d88afec3fa544296bdcd05fda"
)

PARENT_FILES = {
    "threes_rl/J1_NORMAL_START_JOINT_POLICY_VALUE_PROPOSAL.md":
        "26b225c282fb4b58e11484210cf1f45de273714b1b35054f8670081032980bb2",
    "threes_rl/J1_IMPLEMENTATION_READINESS_AUDIT.json":
        "f3e4e8029e159a1db7767164e1623d2e166b139be319d6077d61d7d107a44042",
    "threes_rl/J1_IMPLEMENTATION_PREFLIGHT_CHARTER.md":
        "7f87bc29c5764ccb290b25558f1cfe999083e9fddb089ea652cac9d0b92ab137",
    "threes_rl/j1_joint_policy_value.py":
        "55d9e3206c2905509466c4962006e6cf3426f76647af6d2e60afe674b80c9bfe",
    "tests/test_rl_j1_joint_policy_value.py":
        "e6b169f2d629021f96315380a3cf0ff6eece94a30e5027b1ace4d741499fbfa4",
    "threes_rl/o2_online_option_preflight.py":
        EXPECTED_O2_POWER_SOURCE_SHA256,
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_TEST_EVIDENCE.json":
        "aceab517c4fffc52fe1827468b8408484c0f9ddade594e5200e025d71239137f",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_PROTECTED_ID_DENYLIST.json":
        "0a7be318ebe5281a11ded38f3bbde29745ccb7c3a969585de1788df468fbd763",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_RUNTIME_STORAGE_PROJECTION.json":
        "e023fe04239ceb2d317ab0e26979033db3c2a5c93d4a5016168de442fc97e401",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_LOCK.json":
        "42d1f8d3d6b7bfd62c173a3147ce1eb7dff465aaa92271e7af6bc5fb3c533825",
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json":
        "339e3ef6dcf8c5b3eb1951204d08b97b94b3c4816f993d58509b9b341dc364b1",
}

PARENT_PAYLOADS = {
    "threes_rl/J1_IMPLEMENTATION_READINESS_AUDIT.json": (
        "canonical_payload_sha256",
        "5b6b9a2383296f82b6547bbd46ddc892b486e4b89f4c325aa88f9c8b15944f99",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_TEST_EVIDENCE.json": (
        "test_evidence_payload_sha256",
        "686b1e58daa937076704eec5ebd84b3af6bf2a47d8ec41875fe4901cf5dc988e",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_PROTECTED_ID_DENYLIST.json": (
        "denylist_payload_sha256",
        "22731c89df661419d7ca2bcffdb86240f2ad8974b00e765dd715cf8f4e675add",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_RUNTIME_STORAGE_PROJECTION.json": (
        "projection_payload_sha256",
        "1aaba01b73d53ad10252f0c59c238c8274a9e8f8066a8f3f03f3c0587c6bef0b",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_LOCK.json": (
        "preflight_lock_payload_sha256",
        "e465cec348f987af4c77f062a0e8f8bfa968ddc4ff460b40ba829915791622da",
    ),
    "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
    "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json": (
        "preflight_result_payload_sha256",
        "4d21a092e584d9419a47bef384de164cfc9a8590268a67abefa35afb6b573ce2",
    ),
}

PARENT_PROGRESS_PUBLISHED = {
    "development": {
        "n_pairs": 1_024,
        "worst_power_or_1_50": 0.30078125,
        "mde_80pct_grid": 2.5,
        "mde_grid_power": 0.9453125,
    },
    "confirmation": {
        "n_pairs": 5_120,
        "worst_power_or_1_50": 0.8854166666666666,
        "mde_80pct_grid": 1.5,
    },
}

PHASE_CAPS = {
    "training": {"hours": 72.0, "storage_gib": 24.0},
    "development": {"hours": 24.0, "storage_gib": 8.0},
    "confirmation": {"hours": 120.0, "storage_gib": 16.0},
}

PROSPECTIVE_STREAMS = {
    "train": {
        "rows": TRAIN_ROOTS,
        "logical": 213_000_000_000,
        "deck": 214_000_000_000,
        "slot": 215_000_000_000,
        "candidate_policy": 216_000_000_000,
    },
    "development": {
        "rows": DEVELOPMENT_PAIRS,
        "logical": 217_000_000_000,
        "deck": 218_000_000_000,
        "slot": 219_000_000_000,
        "candidate_policy": 220_000_000_000,
        "control_policy": 221_000_000_000,
    },
    "confirmation": {
        "rows": CONFIRMATION_PAIRS,
        "logical": 222_000_000_000,
        "deck": 223_000_000_000,
        "slot": 224_000_000_000,
        "candidate_policy": 225_000_000_000,
        "control_policy": 226_000_000_000,
    },
}

ZERO_WORK = {
    "execution_markers": 0,
    "j1_or_j1a_streams_reserved": 0,
    "j1_or_j1a_streams_consumed": 0,
    "normal_start_games_generated": 0,
    "scientific_labels": 0,
    "scientific_optimizer_steps": 0,
    "scientific_checkpoints": 0,
    "development_content_reads": 0,
    "confirmation_content_reads": 0,
    "score_or_policy_outcomes": 0,
    "human_session_reads": 0,
    "incumbent_changes": 0,
    "dashboard_changes": 0,
}


class J1AIntegrityError(RuntimeError):
    """An immutable identity or arithmetic contract failed."""


def repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def sha256_path(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    with repo_path(path, repo_root).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def payload_with_hash(
    payload: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop(field, None)
    body[field] = canonical_json_hash(body)
    return body


def verify_payload_hash(payload: Mapping[str, Any], field: str) -> bool:
    body = dict(payload)
    embedded = body.pop(field, None)
    return isinstance(embedded, str) and embedded == canonical_json_hash(body)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    body = payload_with_hash(payload, field)
    serialized = json.dumps(body, indent=2, sort_keys=True) + "\n"
    reloaded = json.loads(serialized)
    if not verify_payload_hash(reloaded, field):
        raise J1AIntegrityError(f"JSON reload instability: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)
    observed = json.loads(path.read_text(encoding="utf-8"))
    if not verify_payload_hash(observed, field):
        raise J1AIntegrityError(f"Written payload hash mismatch: {path}")
    return observed


def _load_json(path: str | Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    payload = json.loads(repo_path(path, repo_root).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise J1AIntegrityError(f"Expected JSON object: {path}")
    return payload


def parent_identity_audit(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    files = {}
    for path, expected in PARENT_FILES.items():
        target = repo_path(path, repo_root)
        observed = sha256_path(target) if target.is_file() else None
        files[path] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": observed == expected,
        }

    payloads = {}
    for path, (field, expected) in PARENT_PAYLOADS.items():
        target = repo_path(path, repo_root)
        try:
            payload = _load_json(target)
            observed = payload.get(field)
            stable = verify_payload_hash(payload, field)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            observed = None
            stable = False
        payloads[path] = {
            "field": field,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "reload_stable": stable,
            "matches": observed == expected and stable,
        }

    parent_result_path = (
        repo_root
        / "threes_rl"
        / "runs"
        / "forensics"
        / "j1_implementation_preflight_v1"
        / "J1_IMPLEMENTATION_PREFLIGHT_RESULT.json"
    )
    parent_result = _load_json(parent_result_path)
    parent_lock = _load_json(parent_result_path.with_name(
        "J1_IMPLEMENTATION_PREFLIGHT_LOCK.json"
    ))
    parent_readiness = _load_json(
        repo_root / "threes_rl" / "J1_IMPLEMENTATION_READINESS_AUDIT.json"
    )
    parent_dir = parent_result_path.parent
    marker_paths = sorted(
        str(path.relative_to(repo_root))
        for path in parent_dir.glob("*MARKER*.json")
    )
    zero_values = parent_result.get("zero_work", {})
    semantic_checks = {
        "all_file_hashes_exact": all(row["matches"] for row in files.values()),
        "all_payload_hashes_exact": all(
            row["matches"] for row in payloads.values()
        ),
        "parent_decision_held": (
            parent_result.get("decision") == "HOLD_J1_IMPLEMENTATION_PREFLIGHT"
            and parent_lock.get("decision")
            == "HOLD_J1_IMPLEMENTATION_PREFLIGHT"
        ),
        "parent_not_scientifically_killed": (
            parent_result.get("kill") == "historical kills unchanged"
        ),
        "parent_zero_work_exact": (
            bool(zero_values)
            and all(value == 0 for value in zero_values.values())
        ),
        "parent_marker_absent": not marker_paths,
        "parent_parameter_count_exact": (
            parent_lock.get("parameter_count") == 411_656
            and parent_readiness["selected_design"]["parameter_count"]
            == 411_656
        ),
        "parent_training_roots_exact": (
            parent_readiness["selected_design"]["training_roots"]
            == TRAIN_ROOTS
        ),
        "parent_power_source_exact": files[
            "threes_rl/o2_online_option_preflight.py"
        ]["matches"],
    }
    return {
        "files": files,
        "payloads": payloads,
        "parent_terminal_decision": parent_result.get("decision"),
        "parent_marker_paths": marker_paths,
        "checks": semantic_checks,
        "passes": all(semantic_checks.values()),
    }


def stream_contract() -> dict[str, Any]:
    parent_denylist = _load_json(
        "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
        "J1_PROTECTED_ID_DENYLIST.json"
    )
    parent_contract = parent_denylist["prospective_stream_contract"]
    parent_intervals = {
        (row["partition"], row["stream_role"]): row
        for row in parent_contract["prospective_intervals"]
    }
    intervals = []
    for role, row in PROSPECTIVE_STREAMS.items():
        rows = int(row["rows"])
        for kind, base in row.items():
            if kind == "rows":
                continue
            intervals.append(
                {
                    "role": role,
                    "kind": kind,
                    "start": int(base),
                    "end_inclusive": int(base) + rows - 1,
                    "rows": rows,
                }
            )
    ordered = sorted(intervals, key=lambda row: row["start"])
    disjoint = all(
        left["end_inclusive"] < right["start"]
        for left, right in zip(ordered, ordered[1:])
    )
    parent_prefixes = all(
        (parent := parent_intervals.get((row["role"], row["kind"])))
        is not None
        and row["start"] == parent["base"]
        and row["end_inclusive"] <= parent["end_inclusive"]
        and row["rows"] <= parent["rows"]
        for row in ordered
    )
    checks = {
        "training_roots_exact": PROSPECTIVE_STREAMS["train"]["rows"]
        == TRAIN_ROOTS,
        "development_pairs_exact": (
            PROSPECTIVE_STREAMS["development"]["rows"]
            == DEVELOPMENT_PAIRS
        ),
        "confirmation_pairs_exact": (
            PROSPECTIVE_STREAMS["confirmation"]["rows"]
            == CONFIRMATION_PAIRS
        ),
        "evaluation_counts_multiple_64": (
            DEVELOPMENT_PAIRS % 64 == 0
            and CONFIRMATION_PAIRS % 64 == 0
        ),
        "total_game_arms_exact": TOTAL_GAME_ARMS == 27_136,
        "all_ranges_above_historical_ceiling": all(
            row["start"] > 212_999_999_999 for row in ordered
        ),
        "all_namespace_ranges_disjoint": disjoint,
        "parent_denylist_contract_passed": (
            parent_denylist.get("passes") is True
            and parent_contract.get("passes") is True
        ),
        "amended_ranges_are_exact_parent_prefixes": parent_prefixes,
        "parent_streams_were_not_reserved_or_consumed": (
            parent_contract.get("streams_reserved") == 0
            and parent_contract.get("streams_consumed") == 0
        ),
        "paired_exogenous_crn_frozen": True,
        "candidate_control_policy_streams_distinct": (
            PROSPECTIVE_STREAMS["development"]["candidate_policy"]
            != PROSPECTIVE_STREAMS["development"]["control_policy"]
            and PROSPECTIVE_STREAMS["confirmation"]["candidate_policy"]
            != PROSPECTIVE_STREAMS["confirmation"]["control_policy"]
        ),
        "streams_not_reserved": True,
        "streams_not_consumed": True,
    }
    payload = {
        "version": f"{VERSION}_stream_contract",
        "prospective_streams": PROSPECTIVE_STREAMS,
        "intervals": ordered,
        "pair_semantics": (
            "logical/deck/slot equal within each candidate-control pair; "
            "policy identities distinct"
        ),
        "inherited_historical_ceiling": 212_999_999_999,
        "parent_denylist": {
            "path": (
                "threes_rl/runs/forensics/"
                "j1_implementation_preflight_v1/"
                "J1_PROTECTED_ID_DENYLIST.json"
            ),
            "file_sha256": PARENT_FILES[
                "threes_rl/runs/forensics/"
                "j1_implementation_preflight_v1/"
                "J1_PROTECTED_ID_DENYLIST.json"
            ],
            "payload_sha256": PARENT_PAYLOADS[
                "threes_rl/runs/forensics/"
                "j1_implementation_preflight_v1/"
                "J1_PROTECTED_ID_DENYLIST.json"
            ][1],
        },
        "prospective_unique_stream_id_count": sum(
            row["rows"] for row in ordered
        ),
        "checks": checks,
        "passes": all(checks.values()),
        "streams_reserved": False,
        "streams_consumed": False,
    }
    payload["contract_sha256"] = canonical_json_hash(payload)
    return payload


def score_power_row(n_pairs: int) -> dict[str, Any]:
    if n_pairs <= 0:
        raise ValueError("Score power requires positive N")
    z975 = SCORE_Z_975
    z80 = SCORE_Z_80
    standardized_effect = (
        math.log1p(SCORE_EFFECT) * math.sqrt(n_pairs) / SCORE_SD
    )
    power = 0.5 * (
        1.0 + math.erf((standardized_effect - z975) / math.sqrt(2.0))
    )
    mde = math.exp((z975 + z80) * SCORE_SD / math.sqrt(n_pairs)) - 1.0
    return {
        "n_pairs": int(n_pairs),
        "paired_log_score_sd": SCORE_SD,
        "meaningful_relative_effect": SCORE_EFFECT,
        "z_0_975": z975,
        "z_0_80": z80,
        "power_at_7pct": power,
        "mde_80pct_relative": mde,
    }


PowerSimulator = Callable[..., Mapping[str, Any]]


def _progression_cell(
    *,
    simulator: PowerSimulator,
    n_pairs: int,
    odds_ratio: float,
    control_rate: float,
    coupling: float,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    row = dict(
        simulator(
            n_roots=n_pairs,
            odds_ratio=odds_ratio,
            base_rates=[control_rate] * STREAM_STRATA,
            coupling=coupling,
            calibration_name=POWER_CALIBRATION_NAME,
            designs=datasets,
            bootstraps=bootstraps,
        )
    )
    required = {
        "n_roots": n_pairs,
        "roots_per_stream_stratum": n_pairs // STREAM_STRATA,
        "base_rates": [control_rate] * STREAM_STRATA,
        "coupling": coupling,
        "true_odds_ratio": odds_ratio,
        "designs": datasets,
        "bootstrap_replicates": bootstraps,
        "gate_point_floor": POWER_POINT_GATE,
        "gate_lower_ci_floor": POWER_LOWER_GATE,
    }
    if any(row.get(key) != value for key, value in required.items()):
        raise J1AIntegrityError("Preserved power simulator contract drift")
    return {
        "n_pairs": n_pairs,
        "control_rate": control_rate,
        "coupling": coupling,
        "true_odds_ratio": odds_ratio,
        "seed": int(row["seed"]),
        "full_gate_power": float(row["full_gate_power"]),
        "monte_carlo_standard_error": float(
            row["monte_carlo_standard_error"]
        ),
        "mean_log_common_or": float(row["mean_log_common_or"]),
    }


def progression_power_summary(
    n_pairs: int,
    *,
    simulator: PowerSimulator = simulate_capability_power,
    odds_ratios: Sequence[float] = ODDS_RATIO_GRID,
    control_rates: Sequence[float] = CONTROL_RATES,
    couplings: Sequence[float] = COUPLINGS,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    if n_pairs <= 0 or n_pairs % STREAM_STRATA:
        raise ValueError("Progression N must be positive and divisible by 8")
    rows = [
        _progression_cell(
            simulator=simulator,
            n_pairs=n_pairs,
            odds_ratio=float(odds_ratio),
            control_rate=float(control_rate),
            coupling=float(coupling),
            datasets=datasets,
            bootstraps=bootstraps,
        )
        for odds_ratio in odds_ratios
        for control_rate in control_rates
        for coupling in couplings
    ]
    worst_by_or = {}
    for odds_ratio in odds_ratios:
        candidates = [
            row
            for row in rows
            if row["true_odds_ratio"] == float(odds_ratio)
        ]
        worst = min(
            candidates,
            key=lambda row: (
                row["full_gate_power"],
                row["control_rate"],
                row["coupling"],
            ),
        )
        worst_by_or[f"{float(odds_ratio):.2f}"] = {
            "power": worst["full_gate_power"],
            "control_rate": worst["control_rate"],
            "coupling": worst["coupling"],
            "seed": worst["seed"],
            "monte_carlo_standard_error": (
                worst["monte_carlo_standard_error"]
            ),
        }
    mde = next(
        (
            float(odds_ratio)
            for odds_ratio in odds_ratios
            if worst_by_or[f"{float(odds_ratio):.2f}"]["power"]
            >= POWER_REQUIRED
        ),
        None,
    )
    return {
        "n_pairs": n_pairs,
        "stream_strata": STREAM_STRATA,
        "datasets_per_cell": datasets,
        "whole_root_bootstraps_per_dataset": bootstraps,
        "calibration_name": POWER_CALIBRATION_NAME,
        "control_rates": list(control_rates),
        "couplings": list(couplings),
        "odds_ratio_grid": list(odds_ratios),
        "point_gate": POWER_POINT_GATE,
        "lower_95_gate": POWER_LOWER_GATE,
        "rows": rows,
        "worst_by_or": worst_by_or,
        "mde_80pct_grid": mde,
    }


def reproduce_parent_progression(
    *,
    simulator: PowerSimulator = simulate_capability_power,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    development = progression_power_summary(
        PARENT_PROGRESS_PUBLISHED["development"]["n_pairs"],
        simulator=simulator,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    confirmation = progression_power_summary(
        PARENT_PROGRESS_PUBLISHED["confirmation"]["n_pairs"],
        simulator=simulator,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    observed = {
        "development": {
            "n_pairs": development["n_pairs"],
            "worst_power_or_1_50": (
                development["worst_by_or"]["1.50"]["power"]
            ),
            "mde_80pct_grid": development["mde_80pct_grid"],
            "mde_grid_power": development["worst_by_or"]["2.50"]["power"],
        },
        "confirmation": {
            "n_pairs": confirmation["n_pairs"],
            "worst_power_or_1_50": (
                confirmation["worst_by_or"]["1.50"]["power"]
            ),
            "mde_80pct_grid": confirmation["mde_80pct_grid"],
        },
    }
    checks = {
        "development_published_cells_exact": (
            observed["development"]
            == PARENT_PROGRESS_PUBLISHED["development"]
        ),
        "confirmation_published_cells_exact": (
            observed["confirmation"]
            == PARENT_PROGRESS_PUBLISHED["confirmation"]
        ),
        "draw_count_exact": datasets == POWER_DATASETS,
        "bootstrap_count_exact": bootstraps == POWER_BOOTSTRAPS,
    }
    return {
        "preserved_source": (
            "threes_rl.o2_online_option_preflight."
            "simulate_capability_power"
        ),
        "preserved_source_sha256": EXPECTED_O2_POWER_SOURCE_SHA256,
        "published": PARENT_PROGRESS_PUBLISHED,
        "observed": observed,
        "development": development,
        "confirmation": confirmation,
        "checks": checks,
        "passes": all(checks.values()),
    }


def progression_power_report(
    *,
    simulator: PowerSimulator = simulate_capability_power,
    datasets: int = POWER_DATASETS,
    bootstraps: int = POWER_BOOTSTRAPS,
) -> dict[str, Any]:
    reproduction = reproduce_parent_progression(
        simulator=simulator,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    if not reproduction["passes"]:
        return {
            "method_reproduction": reproduction,
            "amended": None,
            "accepted": False,
            "decision": "HOLD_METHOD_REPRODUCTION",
        }
    development = progression_power_summary(
        DEVELOPMENT_PAIRS,
        simulator=simulator,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    confirmation = progression_power_summary(
        CONFIRMATION_PAIRS,
        simulator=simulator,
        datasets=datasets,
        bootstraps=bootstraps,
    )
    confirmation_or150 = confirmation["worst_by_or"]["1.50"]["power"]
    checks = {
        "parent_method_reproduced_exactly": reproduction["passes"],
        "development_n_exact": development["n_pairs"] == DEVELOPMENT_PAIRS,
        "confirmation_n_exact": confirmation["n_pairs"]
        == CONFIRMATION_PAIRS,
        "confirmation_or150_power_at_least_80pct": (
            confirmation_or150 >= POWER_REQUIRED
        ),
    }
    return {
        "method_reproduction": reproduction,
        "amended": {
            "development": development,
            "confirmation": confirmation,
        },
        "checks": checks,
        "accepted": all(checks.values()),
        "decision": (
            "ACCEPTED_J1A_PROGRESSION_POWER"
            if all(checks.values())
            else "HOLD_J1A_PROGRESSION_POWER"
        ),
    }


def _phase_projection(
    *,
    phase: str,
    arms: int,
    parent_projection: Mapping[str, Any],
) -> dict[str, Any]:
    timing = parent_projection["fixture_timing"]
    actor_seconds = (
        float(timing["actor_batch"]["p90_seconds"])
        / int(timing["actor_batch_size"])
    )
    simulator_seconds = float(
        timing["simulator_transition"]["p90_seconds"]
    )
    incumbent_seconds = float(
        timing["incumbent_fixed_state_action"]["p90_seconds"]
    )
    update_batch_seconds = float(
        timing["synthetic_forward_backward"]["p90_seconds"]
    )
    decisions = arms * PLANNING_MOVES
    checkpoint_bytes = int(parent_projection["bytes_per_checkpoint"])
    if phase == "training":
        collection = decisions * (actor_seconds + simulator_seconds)
        update_batches = math.ceil(
            decisions * 4 / int(timing["update_batch_size"])
        )
        update_seconds = update_batches * update_batch_seconds
        central_seconds = collection + update_seconds
        round_buffer_bytes = (
            256
            * PLANNING_MOVES
            * int(parent_projection["bytes_per_transition"])
        )
        retained_bytes = (
            arms * int(parent_projection["bytes_per_root_metadata"])
            + 2 * checkpoint_bytes
        )
        peak_bytes = retained_bytes + round_buffer_bytes
    else:
        pair_count = arms // 2
        collection = (
            pair_count
            * PLANNING_MOVES
            * (
                actor_seconds
                + incumbent_seconds
                + 2 * simulator_seconds
            )
        )
        update_seconds = 0.0
        central_seconds = collection
        retained_bytes = (
            arms
            * int(parent_projection["bytes_per_evaluation_root_summary"])
        )
        peak_bytes = retained_bytes + checkpoint_bytes
    cap = PHASE_CAPS[phase]
    margin_seconds = central_seconds * SAFETY_MULTIPLIER
    sensitivity_seconds = central_seconds * (MAX_MOVES / PLANNING_MOVES)
    sensitivity_margin_seconds = sensitivity_seconds * SAFETY_MULTIPLIER
    margin_hours = margin_seconds / 3600.0
    cap_fraction = margin_hours / float(cap["hours"])
    margin_peak_bytes = math.ceil(peak_bytes * SAFETY_MULTIPLIER)
    return {
        "complete_game_arms": arms,
        "planning_decisions": decisions,
        "planning_moves_per_arm": PLANNING_MOVES,
        "collection_seconds": collection,
        "update_seconds": update_seconds,
        "central_hours": central_seconds / 3600.0,
        "hours_with_25pct_margin": margin_hours,
        "runtime_cap_hours": cap["hours"],
        "runtime_cap_fraction_after_margin": cap_fraction,
        "runtime_central_passes": margin_hours < float(cap["hours"]),
        "runtime_at_most_91pct_cap": (
            phase == "training"
            or cap_fraction <= RUNTIME_HEADROOM_MAX_FRACTION
        ),
        "contract_max_5000_move_sensitivity_hours": (
            sensitivity_seconds / 3600.0
        ),
        "contract_max_5000_move_sensitivity_hours_with_25pct_margin": (
            sensitivity_margin_seconds / 3600.0
        ),
        "contract_max_5000_move_sensitivity_runtime_passes": (
            sensitivity_margin_seconds / 3600.0 < float(cap["hours"])
        ),
        "contract_max_sensitivity_is_diagnostic": True,
        "retained_bytes": retained_bytes,
        "peak_bytes": peak_bytes,
        "peak_bytes_with_25pct_margin": margin_peak_bytes,
        "peak_gib_with_25pct_margin": margin_peak_bytes / 1024**3,
        "storage_cap_gib": cap["storage_gib"],
        "storage_passes": (
            margin_peak_bytes / 1024**3 < float(cap["storage_gib"])
        ),
    }


def runtime_storage_projection(
    parent_projection: Mapping[str, Any],
) -> dict[str, Any]:
    parent_counts = {
        "training": TRAIN_ROOTS,
        "development": 2 * 1_024,
        "confirmation": 2 * 5_120,
    }
    reproduced = {
        phase: _phase_projection(
            phase=phase,
            arms=arms,
            parent_projection=parent_projection,
        )
        for phase, arms in parent_counts.items()
    }
    parent_rows = parent_projection["phase_projections"]
    parent_keys = (
        "complete_game_arms",
        "planning_decisions",
        "planning_moves_per_arm",
        "collection_seconds",
        "update_seconds",
        "central_hours",
        "hours_with_25pct_margin",
        "contract_max_5000_move_sensitivity_hours",
        "contract_max_5000_move_sensitivity_hours_with_25pct_margin",
        "contract_max_5000_move_sensitivity_runtime_passes",
        "contract_max_sensitivity_is_diagnostic",
        "retained_bytes",
        "peak_bytes",
        "peak_bytes_with_25pct_margin",
        "peak_gib_with_25pct_margin",
        "runtime_cap_hours",
        "runtime_central_passes",
        "storage_cap_gib",
        "storage_passes",
    )
    reproduction_checks = {
        phase: all(
            reproduced[phase][key] == parent_rows[phase][key]
            for key in parent_keys
        )
        for phase in parent_counts
    }

    amended_counts = {
        "training": TRAIN_ROOTS,
        "development": 2 * DEVELOPMENT_PAIRS,
        "confirmation": 2 * CONFIRMATION_PAIRS,
    }
    amended = {
        phase: _phase_projection(
            phase=phase,
            arms=arms,
            parent_projection=parent_projection,
        )
        for phase, arms in amended_counts.items()
    }
    checks = {
        "sealed_parent_projection_reproduced_exactly": all(
            reproduction_checks.values()
        ),
        "fixture_bytes_reused": True,
        "fixture_not_retimed": True,
        "total_game_arms_exact": (
            sum(row["complete_game_arms"] for row in amended.values())
            == TOTAL_GAME_ARMS
        ),
        "training_original_runtime_cap_passes": (
            amended["training"]["runtime_central_passes"]
        ),
        "training_original_storage_cap_passes": (
            amended["training"]["storage_passes"]
        ),
        "development_original_caps_pass": (
            amended["development"]["runtime_central_passes"]
            and amended["development"]["storage_passes"]
        ),
        "confirmation_original_caps_pass": (
            amended["confirmation"]["runtime_central_passes"]
            and amended["confirmation"]["storage_passes"]
        ),
        "development_at_most_91pct_runtime_cap": (
            amended["development"]["runtime_at_most_91pct_cap"]
        ),
        "confirmation_at_most_91pct_runtime_cap": (
            amended["confirmation"]["runtime_at_most_91pct_cap"]
        ),
        "5000_move_sensitivity_reported_with_margin": all(
            "contract_max_5000_move_sensitivity_hours_with_25pct_margin"
            in row
            for row in amended.values()
        ),
    }
    return {
        "version": f"{VERSION}_runtime_storage",
        "sealed_parent_projection_file_sha256": PARENT_FILES[
            "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
            "J1_RUNTIME_STORAGE_PROJECTION.json"
        ],
        "sealed_parent_projection_payload_sha256": PARENT_PAYLOADS[
            "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
            "J1_RUNTIME_STORAGE_PROJECTION.json"
        ][1],
        "fixture_timing": parent_projection["fixture_timing"],
        "parent_reproduction": {
            "phase_projections": reproduced,
            "checks": reproduction_checks,
            "passes": all(reproduction_checks.values()),
        },
        "amended_phase_projections": amended,
        "safety_multiplier": SAFETY_MULTIPLIER,
        "central_planning_moves": PLANNING_MOVES,
        "max_moves_sensitivity": MAX_MOVES,
        "runtime_headroom_max_fraction": RUNTIME_HEADROOM_MAX_FRACTION,
        "checks": checks,
        "passes": all(checks.values()),
        "fixture_retimed": False,
    }


def arithmetic_report(
    *,
    simulator: PowerSimulator = simulate_capability_power,
) -> dict[str, Any]:
    parent_projection = _load_json(
        "threes_rl/runs/forensics/j1_implementation_preflight_v1/"
        "J1_RUNTIME_STORAGE_PROJECTION.json"
    )
    score = {
        "parent_reproduction": {
            "development": score_power_row(1_024),
            "confirmation": score_power_row(5_120),
        },
        "amended": {
            "development": score_power_row(DEVELOPMENT_PAIRS),
            "confirmation": score_power_row(CONFIRMATION_PAIRS),
        },
    }
    parent_readiness = _load_json(
        "threes_rl/J1_IMPLEMENTATION_READINESS_AUDIT.json"
    )
    parent_score = parent_readiness["power_contract"]["score"]
    score_reproduction = {
        role: score["parent_reproduction"][role]
        ["mde_80pct_relative"]
        == parent_score[role]["mde_80pct_relative"]
        and score["parent_reproduction"][role]["power_at_7pct"]
        == parent_score[role]["power_at_7pct"]
        for role in ("development", "confirmation")
    }
    score["checks"] = {
        "parent_score_method_reproduced_exactly": all(
            score_reproduction.values()
        ),
        "confirmation_power_at_least_95pct": (
            score["amended"]["confirmation"]["power_at_7pct"]
            >= SCORE_REQUIRED_POWER
        ),
        "confirmation_mde_below_5_5pct": (
            score["amended"]["confirmation"]["mde_80pct_relative"]
            < SCORE_MDE_MAX
        ),
        "development_is_permissive_screen": True,
    }
    score["passes"] = all(score["checks"].values())
    progression = progression_power_report(simulator=simulator)
    runtime_storage = runtime_storage_projection(parent_projection)
    checks = {
        "score": score["passes"],
        "progression_method_reproduced": progression[
            "method_reproduction"
        ]["passes"],
        "progression_amended_accepted": progression["accepted"],
        "runtime_storage": runtime_storage["passes"],
    }
    return {
        "version": f"{VERSION}_arithmetic",
        "score": score,
        "progression": progression,
        "runtime_storage": runtime_storage,
        "checks": checks,
        "passes": all(checks.values()),
        "new_policy_outcomes_opened": 0,
        "fixture_retimed": False,
    }


def test_evidence_payload(
    *,
    focused_command: str,
    focused_passed: int,
    parent_command: str,
    parent_passed: int,
    namespace_fresh_before_evidence: bool,
) -> dict[str, Any]:
    checks = {
        "focused_command_exact": focused_command == FOCUSED_TEST_COMMAND,
        "focused_tests_exact": focused_passed == FOCUSED_TEST_COUNT,
        "parent_command_exact": parent_command == PARENT_TEST_COMMAND,
        "parent_j1_tests_exact": parent_passed == PARENT_TEST_COUNT,
        "namespace_fresh_before_evidence": namespace_fresh_before_evidence,
        "amendment_exists": AMENDMENT_PATH.is_file(),
        "runner_exists": RUNNER_PATH.is_file(),
        "tests_exist": TEST_PATH.is_file(),
    }
    return {
        "version": f"{VERSION}_test_evidence",
        "source_identities": {
            str(AMENDMENT_PATH.relative_to(REPO_ROOT)): sha256_path(
                AMENDMENT_PATH
            ),
            str(RUNNER_PATH.relative_to(REPO_ROOT)): sha256_path(RUNNER_PATH),
            str(TEST_PATH.relative_to(REPO_ROOT)): sha256_path(TEST_PATH),
        },
        "commands": [
            {
                "kind": "focused_j1a",
                "command": focused_command,
                "passed": focused_passed,
            },
            {
                "kind": "parent_j1_regression",
                "command": parent_command,
                "passed": parent_passed,
            },
        ],
        "checks": checks,
        "passes": all(checks.values()),
        "zero_work": ZERO_WORK,
    }


def write_test_evidence(
    *,
    output_dir: Path,
    focused_command: str,
    focused_passed: int,
    parent_command: str,
    parent_passed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"J1a namespace must be fresh before evidence: {output_dir}"
        )
    payload = test_evidence_payload(
        focused_command=focused_command,
        focused_passed=focused_passed,
        parent_command=parent_command,
        parent_passed=parent_passed,
        namespace_fresh_before_evidence=True,
    )
    if not payload["passes"]:
        raise J1AIntegrityError("Test evidence does not match frozen commands")
    return write_immutable_json(
        output_dir / TEST_EVIDENCE_NAME,
        payload,
        field="test_evidence_payload_sha256",
    )


def verify_test_evidence(output_dir: Path) -> dict[str, Any]:
    path = output_dir / TEST_EVIDENCE_NAME
    payload = _load_json(path)
    expected_sources = {
        str(AMENDMENT_PATH.relative_to(REPO_ROOT)): sha256_path(
            AMENDMENT_PATH
        ),
        str(RUNNER_PATH.relative_to(REPO_ROOT)): sha256_path(RUNNER_PATH),
        str(TEST_PATH.relative_to(REPO_ROOT)): sha256_path(TEST_PATH),
    }
    checks = {
        "payload_reload_stable": verify_payload_hash(
            payload, "test_evidence_payload_sha256"
        ),
        "source_identities_current": payload.get("source_identities")
        == expected_sources,
        "test_checks_passed": payload.get("passes") is True,
        "zero_work_exact": payload.get("zero_work") == ZERO_WORK,
    }
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "file_sha256": sha256_path(path),
        "payload_sha256": payload.get("test_evidence_payload_sha256"),
        "checks": checks,
        "passes": all(checks.values()),
    }


def zero_work_audit(output_dir: Path) -> dict[str, Any]:
    allowed_before_prepare = {TEST_EVIDENCE_NAME}
    existing = (
        sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        )
        if output_dir.exists()
        else []
    )
    forbidden_markers = (
        sorted(path.name for path in output_dir.glob("*MARKER*.json"))
        if output_dir.exists()
        else []
    )
    checks = {
        "namespace_contains_only_test_evidence": set(existing)
        <= allowed_before_prepare,
        "test_evidence_exists": existing == [TEST_EVIDENCE_NAME],
        "no_execution_marker": not forbidden_markers,
        "all_forbidden_work_zero": all(
            value == 0 for value in ZERO_WORK.values()
        ),
    }
    return {
        "existing_files_before_prepare": existing,
        "forbidden_marker_paths": forbidden_markers,
        "counters": ZERO_WORK,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _decision(
    *,
    parent: Mapping[str, Any],
    arithmetic: Mapping[str, Any],
    streams: Mapping[str, Any],
    evidence: Mapping[str, Any],
    zero_work: Mapping[str, Any],
    operational: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    method_reproduced = bool(
        arithmetic["progression"]["method_reproduction"]["passes"]
    )
    integrity_checks = {
        "parent_identities": bool(parent["passes"]),
        "stream_schema": bool(streams["passes"]),
        "test_evidence": bool(evidence["passes"]),
        "zero_work": bool(zero_work["passes"]),
        "score_method_reproduced": bool(
            arithmetic["score"]["checks"][
                "parent_score_method_reproduced_exactly"
            ]
        ),
        "runtime_method_reproduced": bool(
            arithmetic["runtime_storage"]["checks"][
                "sealed_parent_projection_reproduced_exactly"
            ]
        ),
    }
    gate_checks = {
        "score_power_and_mde": bool(arithmetic["score"]["passes"]),
        "progression_power": bool(
            arithmetic["progression"]["accepted"]
        ),
        "runtime_storage_headroom": bool(
            arithmetic["runtime_storage"]["passes"]
        ),
        "operational": bool(operational["passes"]),
    }
    checks = {
        "method_reproduction": method_reproduced,
        **integrity_checks,
        **gate_checks,
    }
    if not method_reproduced:
        decision = "HOLD_METHOD_REPRODUCTION"
    elif not all(integrity_checks.values()):
        decision = "KILL_J1A_AMENDMENT_INTEGRITY"
    elif not all(gate_checks.values()):
        decision = "HOLD_J1A_COST_POWER_AMENDMENT"
    else:
        decision = "READY_J1A_COST_POWER_AMENDMENT"
    return decision, checks


def prepare(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    zero = zero_work_audit(output_dir)
    evidence = verify_test_evidence(output_dir)
    parent = parent_identity_audit()
    if not parent["passes"]:
        raise J1AIntegrityError(
            "Parent J1 identity audit failed before arithmetic"
        )
    streams = stream_contract()
    arithmetic = arithmetic_report()
    operational = j1.operational_audit(output_dir=output_dir)
    decision, checks = _decision(
        parent=parent,
        arithmetic=arithmetic,
        streams=streams,
        evidence=evidence,
        zero_work=zero,
        operational=operational,
    )
    arithmetic_payload = {
        **arithmetic,
        "parent_identity_audit": parent,
        "stream_contract": streams,
        "zero_work": ZERO_WORK,
    }
    arithmetic_written = write_immutable_json(
        output_dir / ARITHMETIC_NAME,
        arithmetic_payload,
        field="arithmetic_payload_sha256",
    )
    arithmetic_identity = {
        "path": ARITHMETIC_NAME,
        "file_sha256": sha256_path(output_dir / ARITHMETIC_NAME),
        "payload_sha256": arithmetic_written[
            "arithmetic_payload_sha256"
        ],
    }
    lock_payload = {
        "version": f"{VERSION}_preflight_lock",
        "bound_output_dir": str(output_dir.resolve()),
        "amendment": {
            "path": str(AMENDMENT_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_path(AMENDMENT_PATH),
        },
        "runner": {
            "path": str(RUNNER_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_path(RUNNER_PATH),
        },
        "tests": {
            "path": str(TEST_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_path(TEST_PATH),
        },
        "parent_files": PARENT_FILES,
        "parent_payloads": {
            path: {"field": field, "sha256": expected}
            for path, (field, expected) in PARENT_PAYLOADS.items()
        },
        "test_evidence": evidence,
        "arithmetic": arithmetic_identity,
        "counts": {
            "training_roots": TRAIN_ROOTS,
            "development_pairs": DEVELOPMENT_PAIRS,
            "confirmation_pairs": CONFIRMATION_PAIRS,
            "total_complete_game_arms": TOTAL_GAME_ARMS,
        },
        "power_contract": {
            "datasets_per_cell": POWER_DATASETS,
            "whole_root_bootstraps_per_dataset": POWER_BOOTSTRAPS,
            "control_rates": list(CONTROL_RATES),
            "couplings": list(COUPLINGS),
            "odds_ratio_grid": list(ODDS_RATIO_GRID),
            "calibration_name": POWER_CALIBRATION_NAME,
            "source_sha256": EXPECTED_O2_POWER_SOURCE_SHA256,
        },
        "operational_audit": operational,
        "checks": checks,
        "decision": decision,
        "passes": decision == "READY_J1A_COST_POWER_AMENDMENT",
        "execution_command_defined": False,
        "marker_defined": False,
        "zero_work": ZERO_WORK,
    }
    lock_written = write_immutable_json(
        output_dir / PREFLIGHT_LOCK_NAME,
        lock_payload,
        field="preflight_lock_payload_sha256",
    )
    lock_identity = {
        "path": PREFLIGHT_LOCK_NAME,
        "file_sha256": sha256_path(output_dir / PREFLIGHT_LOCK_NAME),
        "payload_sha256": lock_written["preflight_lock_payload_sha256"],
    }
    result_payload = {
        "version": f"{VERSION}_preflight_result",
        "decision": decision,
        "continue": decision == "READY_J1A_COST_POWER_AMENDMENT",
        "hold": "all J1/J1a execution and science",
        "kill": (
            "historical kills unchanged; J1/J1a not scientifically killed"
        ),
        "promote": False,
        "next_authority": (
            "research-lead review for a separately frozen J1 execution surface"
            if decision == "READY_J1A_COST_POWER_AMENDMENT"
            else "no J1/J1a execution"
        ),
        "checks": checks,
        "test_evidence": evidence,
        "arithmetic": arithmetic_identity,
        "preflight_lock": lock_identity,
        "zero_work": ZERO_WORK,
    }
    result_written = write_immutable_json(
        output_dir / PREFLIGHT_RESULT_NAME,
        result_payload,
        field="preflight_result_payload_sha256",
    )
    return {
        "decision": decision,
        "output_dir": str(output_dir),
        "test_evidence": evidence,
        "arithmetic": arithmetic_identity,
        "preflight_lock": lock_identity,
        "preflight_result": {
            "path": PREFLIGHT_RESULT_NAME,
            "file_sha256": sha256_path(output_dir / PREFLIGHT_RESULT_NAME),
            "payload_sha256": result_written[
                "preflight_result_payload_sha256"
            ],
        },
        "checks": checks,
        "zero_work": ZERO_WORK,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence = subparsers.add_parser("write-test-evidence")
    evidence.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    evidence.add_argument("--focused-command", required=True)
    evidence.add_argument("--focused-passed", type=int, required=True)
    evidence.add_argument("--parent-command", required=True)
    evidence.add_argument("--parent-passed", type=int, required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "write-test-evidence":
        payload = write_test_evidence(
            output_dir=args.output_dir,
            focused_command=args.focused_command,
            focused_passed=args.focused_passed,
            parent_command=args.parent_command,
            parent_passed=args.parent_passed,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare":
        payload = prepare(args.output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

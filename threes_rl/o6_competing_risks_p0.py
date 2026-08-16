"""Read-only source preparation contract for O6 competing-risks P0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "o6_competing_risks_p0_source_preparation_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

CHARTER_PATH = Path("threes_rl/O6_COMPETING_RISKS_P0_CHARTER.md")
PROPOSAL_PATH = Path("threes_rl/O6_COMPETING_RISKS_EVENT_PROPOSAL.md")
O5_AUDIT_PATH = Path("threes_rl/O5_TRUE_H40_CENSOR_MECHANISM_AUDIT.md")
RUNNER_PATH = Path("threes_rl/o6_competing_risks_p0.py")
TEST_PATH = Path("tests/test_rl_o6_competing_risks_p0.py")
OUTPUT_DIR = Path("threes_rl/runs/forensics/o6_competing_risks_p0_v1")
TEST_EVIDENCE_PATH = Path(
    "threes_rl/runs/forensics/"
    "o6_competing_risks_p0_test_evidence_v1.json"
)

FAMILY_ORDER = (
    "o6_corner2",
    "o6_expectimax2",
    "o6_parent_mc1000",
    "o6_replaycal",
)
FAMILY_SIGNATURES = {
    "o6_corner2": (
        "4be4214166f40ddaaac5af499cb1e1d08d992b0a90bb680cfcb7cab04d217043"
    ),
    "o6_expectimax2": (
        "2ad642cdca7739cc73af4f570de5054c422815f9a7d8f93a2619921b46b74b38"
    ),
    "o6_parent_mc1000": (
        "e43dc11f3220557d7f9aef228db96dc6f06f49b26300d5a4128ea00bf8ba2064"
    ),
    "o6_replaycal": (
        "e07c566b55d86a889ab7ca54d01c00c9b6cdf808fdb1627f70596bd829fdeab3"
    ),
}
ACCEPTED_POLICY_LOCK_SHA256 = (
    "6b0384d9fedfc8f560853a050c28750194ec9c9d3d36cf2d9d7fd47a9a423ea0"
)
TARGET_ORDER = (48, 96, 192)
ROLE_ORDER = ("train", "development", "untouched_mechanism")
POWER_ROOT_COUNTS = (192, 256, 384, 512)
POWER_OR_GRID = (1.25, 1.35, 1.50, 1.75, 2.00)
POWER_RHO_GRID = (0.05, 0.15, 0.25)
POWER_DATASETS = 4_096
POWER_BOOTSTRAPS = 4_096
POWER_DATASET_BATCH_SIZE = 16
POWER_BOOTSTRAP_BATCH_SIZE = 64
POWER_REPEATS_PER_ARM = 8
POWER_BASE_RATE = 188.0 / 1_152.0
POWER_POINT_GATE = 1.25
POWER_LOWER_GATE = 1.0
POWER_REQUIRED = 0.80
POWER_STRATA = (
    "T48_aligned",
    "T48_unaligned",
    "T96_aligned",
    "T96_unaligned",
    "T192_aligned",
    "T192_unaligned",
)

ROLE_COUNTS_BY_UNTOUCHED_N = {
    192: {"train": 384, "development": 96, "untouched_mechanism": 192},
    256: {"train": 512, "development": 128, "untouched_mechanism": 256},
    384: {"train": 768, "development": 192, "untouched_mechanism": 384},
    512: {"train": 1_024, "development": 256, "untouched_mechanism": 512},
}

STREAM_WINDOWS = {
    "label_learning": {
        "logical_seed": 197_000_000_000,
        "deck_stream_id": 198_000_000_000,
        "slot_stream_id": 199_000_000_000,
        "policy_stream_id": 200_000_000_000,
    },
    "mechanism": {
        "logical_seed": 201_000_000_000,
        "deck_stream_id": 202_000_000_000,
        "slot_stream_id": 203_000_000_000,
        "policy_stream_id": 204_000_000_000,
    },
    "normal_development": {
        "logical_seed": 205_000_000_000,
        "deck_stream_id": 206_000_000_000,
        "slot_stream_id": 207_000_000_000,
        "policy_stream_id": 208_000_000_000,
    },
    "confirmation": {
        "logical_seed": 209_000_000_000,
        "deck_stream_id": 210_000_000_000,
        "slot_stream_id": 211_000_000_000,
        "policy_stream_id": 212_000_000_000,
    },
}
STREAM_FIELDS = (
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)
STREAM_WINDOW_SIZE = 1_000_000

EXPECTED_DEPENDENCY_SHA256 = {
    str(PROPOSAL_PATH): (
        "a3ff3bebc7251cbb6dd60acb5c594cdd9ab427cebb667fb56b8ab1b04ddfc770"
    ),
    str(O5_AUDIT_PATH): (
        "80de4e1ad8cbf17fe5bdda10874b38219df0deb0c0928752f27a8fe691ec9a76"
    ),
    "threes_rl/o4_domain_safe_pair_option.py": (
        "95a4da48fb7550e87b09e1f1594cdbdc062a52c7df544b7445b5e58878c87f41"
    ),
    "threes_rl/o4_power_contract.py": (
        "16e2c26c9e1f2b176937f1a0546604b878d45875b4c29dbc83a441588f7fc5cd"
    ),
    "threes_rl/o5_four_family_p0.py": (
        "f0ffcc17578581b6e4783e63beef28e59ffab16676ddbd84126127fab47bcff6"
    ),
    "threes_rl/o5_training_v2.py": (
        "37e0a20d2437f09ef7efe1073573f7f53c4ed8ae0267560192e7892164e956ea"
    ),
    "threes_rl/o3_p0_preflight.py": (
        "3a34f427f3929fb8a1383406822aa69990f4c3b7e0a5009cac3270dcb3d25213"
    ),
    "threes_rl/o4_p0_preflight_v2.py": (
        "edf8f356263b1cab09dffd49389ad755acbd0f70d704de4d43b0f27d7e9a5537"
    ),
    "threes_rl/sim.py": (
        "67e7a245c05e59367402095ad018122fb4cb1ef08664bf28bf4bc03a02a73072"
    ),
    "threes_rl/replay_provenance.py": (
        "2867cdd23973a4c5464905bf05373a6a0ae3e4439bfd9ac9de1e30892848e992"
    ),
    "threes_rl/g1r_acquire.py": (
        "73ba88103024e6cf62ba4418d88a9bbe71cf42aafc1b911ef39818647f655d6a"
    ),
    "threes_rl/g1r_acquire_v2_qd5.py": (
        "f195026041e25aeb22ffc72cc57c49d1da96a1af3dfa9fa9180e31345a13d776"
    ),
    "threes_rl/g1r_qd_admission_v2.py": (
        "191c612d183832bc79ec376322a5c15eae92512360231b6596b307240532c51b"
    ),
    "threes_rl/eval.py": (
        "df0a558014583fcfd24fd8ddf48988e375ad9a6fc5199d35311c40d8b6a3f705"
    ),
    "threes_rl/ntuple.py": (
        "bdd38ec758ca1786b67a7550b3a2792cbd517176ad99e4df7c5ddd2584953789"
    ),
    "threes_rl/expectimax.py": (
        "98a7f0d05437d01555ea37d21211fa36d7260cba84456b0fb08799472b26ec14"
    ),
    "threes_rl/current_incumbent_policy.txt": (
        "d85a91576b8dc0ad80c2ed041dd1a0d62498eac9edb48445cb73233bb5454dd4"
    ),
    "threes_rl/top_replay_playlist.py": (
        "6eab07a34f4ec7849329f22129e8e0b6fc911231857e5dbd0838ec3278b0b256"
    ),
}

CORE_GOVERNANCE_SHA256 = {
    "threes_rl/runs/forensics/o3_event_acquisition_recovery_v1/"
    "O3_RECOVERY_UNION_MANIFEST.json": (
        "02ea2c5be8823de775f56b7267f9c8371d26efc53897115b25733f8ef4527311"
    ),
    "threes_rl/runs/forensics/o3_event_acquisition_recovery_v1/"
    "O3_RECOVERY_SELECTED_ROOTS.json": (
        "9ca8280c82c18d7eb9efb72b7d5c7974d4fdec84549b0607c1f41ded3f23f049"
    ),
    "threes_rl/runs/forensics/o3_event_option_p0_v1/"
    "O3_P0_STREAM_MANIFEST.json": (
        "94e7b0dfe83e568b4e9686dd3ee44cc70739c0312349fe36a05bb6df80c77225"
    ),
    "threes_rl/runs/forensics/o3_event_option_p0_v1/"
    "O3_P0_POLICY_AUDIT.json": (
        "2b498ce5bc22f54f6286e114f3212758e911a1ac7a651da2c3095db42dea0e60"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v1/O4_P0_OPENED.json": (
        "7f84bbd9679b9d6294a0530b47b5ba01749426191a1a3f509bf38a48114723b6"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v1/"
    "O4_P0_V1_ENGINEERING_HOLD.json": (
        "17be1eb2c5ecf0be1a7331779e5eab7cc3159eb760d50d4f4b7aacdf395332e8"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/"
    "O4_P0_V2_OPENED.json": (
        "9d9f032f61fa637941d677e788dcb7d2dcec70179a7ea9a2fafe128af73336da"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/"
    "O4_P0_V2_RESULT.json": (
        "897cac07ce2625f5616690f0a4611e11948e6ca58a55b828ee43f92b493893cd"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/"
    "O4_P0_SELECTED_ROOTS.json": (
        "f3b0a0afb3344e3413e5f63bf367c86aab66bc619d8ec736fbc7313090182ab3"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/"
    "O4_P0_STREAM_MANIFEST.json": (
        "24c94fe8898847a6b54676aec6d5e78511bf687ec18dc5d410a9194a0bde6828"
    ),
    "threes_rl/runs/forensics/o4_domain_safe_p0_v2/"
    "O4_P0_POLICY_AUDIT.json": (
        "366f92f9f0b28bceb287528c4f8c3fd28b2b535f54f0586744368d034640b260"
    ),
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/"
    "O5_P0_OPENED.json": (
        "902df97928d2b393c8819887717c213b831f8321ac0270a0761633737b668c13"
    ),
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/"
    "O5_P0_RESULT.json": (
        "b2ca5368dd6f29debfd0fb0e4c86005c9bae7b92d736ebc5750c5ec71f97a96f"
    ),
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/"
    "O5_P0_SELECTED_ROOTS.json": (
        "d6220ee3ebfe799d78cba128be816e607947a225f7b6ad8add0cc2aad91abad8"
    ),
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/"
    "O5_P0_STREAM_MANIFEST.json": (
        "bf114875f9ff24f4456fdf85aa8fcba86f4c9d7eadc3df43f6e931a50eb35186"
    ),
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/"
    "O5_P0_POLICY_AUDIT.json": (
        "283acb7c2d44dd0c4eea776db8a87dfe810297c0be5cf888e18bed2956a5ff8d"
    ),
    "threes_rl/runs/forensics/o5_four_family_domain_safe_p0_v1/"
    "O5_P0_COLLISION_AUDIT.json": (
        "21eb13c8a7c1b3ee7f5f110540a06e45dccc58bf1a79f300b490b806d6d1420e"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "O5_TRAINING_V2_OPENED.json": (
        "534151b8514336db2fe8d5946c8c66acb42ec1b0931a6d41c9ccf66ed9578cd8"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "training_result.json": (
        "74ac4ca9f375ff93e2fed5dfa5c2154a7b4fcc682654539e05cc67cc4a515e05"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "selected_root_manifest.json": (
        "2d6a75cddd9f4e8bfa84e8e1516b628b05b33bb75ec86c8c1025d55a8057317e"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "learning_task_manifest.json": (
        "7545c020ac7e99c484cdf2c4bcc133658de7eb1214d0915163b057aeb8fd6318"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "label_support_report.json": (
        "bd905b62f05c95c42dc36336f9133d6d80044687476be396161d43ada10a7a94"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "checkpoint_quarantine.json": (
        "96a5336f3a9c37dad56447ceedf9481cd39fe0d6f896effa5f47b07b9c461ece"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "preflight_lock.json": (
        "bfe02142889f56acafc02f5d8885d9e0be932cd83f429819b0c3ae1ac3132b2b"
    ),
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/"
    "preflight_result.json": (
        "a18dc6aa5a27dd22dbd45c2d560d9874f76bb3356e521ecb1523a40d2ed03468"
    ),
    "threes_rl/runs/replays/top3/manifest.json": (
        "48e1294bbe26e2c640ad3ee165af146e86d061c29506d266abdcccbf891afc01"
    ),
}

PROTECTED_DISCOVERY_ROOTS = (
    "threes_rl/runs/eval_manifests",
    "threes_rl/runs/eval_artifacts",
    "threes_rl/runs/forensics",
    "threes_rl/runs/continuations",
    "threes_rl/runs/replays",
    "threes_rl/runs/human_diagnostics",
)
PROTECTED_DISCOVERY_EXTENSIONS = (".json", ".jsonl", ".csv")
PROTECTED_DISCOVERY_PATTERNS = tuple(
    f"{root}/**/*{extension}"
    for root in PROTECTED_DISCOVERY_ROOTS
    for extension in PROTECTED_DISCOVERY_EXTENSIONS
) + tuple(
    f"threes_rl/runs/dashboard/*{extension}"
    for extension in PROTECTED_DISCOVERY_EXTENSIONS
)
GOVERNANCE_FILENAME_TOKENS = (
    "manifest",
    "lock",
    "marker",
    "opened",
    "result",
    "seal",
    "audit",
    "selection",
    "selected",
    "roots",
    "streams",
    "collision",
    "retention",
    "attempt",
    "completion",
    "completed",
    "runtime",
    "task",
    "config",
    "preflight",
)
HASH_ONLY_IDENTITY_COMPANION_REQUIRED = True
FORBIDDEN_BODY_PREFIXES = (
    "threes_rl/runs/human_diagnostics",
    "threes_rl/runs/forensics/o3_option_training_v1/episodes",
    "threes_rl/runs/forensics/o3_option_training_v1/checkpoints",
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/episodes",
    "threes_rl/runs/forensics/o5_domain_safe_training_v2/checkpoints",
)
LIVE_SUMMARY_PATHS = (
    "threes_rl/runs/dashboard/dashboard.json",
    "threes_rl/runs/dashboard/score_trends.json",
)
PROTECTED_IDENTITY_FIELDS = (
    "ancestry",
    "ancestry_id",
    "root",
    "root_cluster",
    "source_replay_sha256",
    "replay_sha256",
    "state_sha1",
    "state_sha256",
    "logical_seed",
    "deck_stream_id",
    "slot_stream_id",
    "policy_stream_id",
)

RISK_SCHEMA = {
    "version": "o6_competing_risks_source_v1",
    "horizon": 40,
    "status_order": ("success", "failure", "live"),
    "row_order": (
        "t",
        "time_fraction",
        "safe_merge_event",
        "competing_failure_event",
        "live_after_transition",
    ),
    "row_mapping": {
        "success": (1.0, 0.0, 0.0),
        "failure": (0.0, 1.0, 0.0),
        "live": (0.0, 0.0, 1.0),
    },
    "success_time_bands": ((1, 10), (11, 20), (21, 40)),
    "short_all_live_invalid": True,
    "h40_all_live_is_administrative_censor": True,
    "censor_is_not_event_class": True,
}

SOURCE_STATE_SCHEMA = {
    "version": "o6_current_state_whitelist_v1",
    "top_level_fields": (
        "board",
        "preview",
        "tile_cycle",
        "move_count",
        "game_over",
    ),
    "tile_cycle_fields": (
        "small_counts",
        "small_pos",
        "small_seen_total",
        "span_small_pos",
        "large_pending",
    ),
    "forbidden_fields": (
        "score",
        "final_score",
        "max_tile",
        "legal_actions",
        "move",
        "action",
        "outcome",
        "milestone",
    ),
}

MIN_FREE_GIB = 100.0
TARGET_FREE_GIB = 120.0
MINIMUM_FAMILIES = 4
MAX_FAMILY_SHARE = 0.40


class O6ContractError(RuntimeError):
    """Fail-closed preparation-contract error."""


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


RISK_SCHEMA_SHA256 = canonical_json_hash(RISK_SCHEMA)
SOURCE_STATE_SCHEMA_SHA256 = canonical_json_hash(SOURCE_STATE_SCHEMA)
PROTECTED_CONTRACT_SHA256 = canonical_json_hash(
    {
        "patterns": PROTECTED_DISCOVERY_PATTERNS,
        "governance_tokens": GOVERNANCE_FILENAME_TOKENS,
        "forbidden_body_prefixes": FORBIDDEN_BODY_PREFIXES,
        "live_summary_paths": LIVE_SUMMARY_PATHS,
        "identity_fields": PROTECTED_IDENTITY_FIELDS,
        "hash_only_identity_companion_required": (
            HASH_ONLY_IDENTITY_COMPANION_REQUIRED
        ),
    }
)
POWER_CONTRACT_SHA256 = canonical_json_hash(
    {
        "root_counts": POWER_ROOT_COUNTS,
        "or_grid": POWER_OR_GRID,
        "rho_grid": POWER_RHO_GRID,
        "datasets": POWER_DATASETS,
        "bootstraps": POWER_BOOTSTRAPS,
        "dataset_batch_size": POWER_DATASET_BATCH_SIZE,
        "bootstrap_batch_size": POWER_BOOTSTRAP_BATCH_SIZE,
        "repeats_per_arm": POWER_REPEATS_PER_ARM,
        "base_rate": POWER_BASE_RATE,
        "point_gate": POWER_POINT_GATE,
        "lower_gate": POWER_LOWER_GATE,
        "required_power": POWER_REQUIRED,
        "strata": POWER_STRATA,
    }
)
EXPECTED_RISK_SCHEMA_SHA256 = (
    "b46e3cd785902bb5753c9066defe8fbcad3fc9bcef6666f2ad72833067042cac"
)
EXPECTED_SOURCE_STATE_SCHEMA_SHA256 = (
    "781d640f9f6eddf1a7ed75551f443651306308875b6fe265b7af19c7278b8671"
)
EXPECTED_PROTECTED_CONTRACT_SHA256 = (
    "4f6386e183bf7c7a3d2106dc6dfdbbc16044878b91241723962e9b65ea21d790"
)
EXPECTED_POWER_CONTRACT_SHA256 = (
    "a3966cbbae5981b3b180055833b562486eee5f93460ecfef6a44a067a73dfad4"
)


def _repo_path(relative: str | Path) -> Path:
    return REPO_ROOT / Path(relative)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_contract(
    expected: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative, frozen_hash in expected.items():
        path = repo_root / relative
        resolved = path.resolve()
        try:
            resolved.relative_to(repo_root.resolve())
            inside_repo = True
        except ValueError:
            inside_repo = False
        exists = path.exists()
        regular = exists and path.is_file() and not path.is_symlink()
        actual = sha256_path(path) if regular else None
        files[relative] = {
            "expected_sha256": frozen_hash,
            "actual_sha256": actual,
            "exists": exists,
            "regular_nonsymlink": regular,
            "inside_repo": inside_repo,
            "matches": regular and inside_repo and actual == frozen_hash,
            "bytes": int(path.stat().st_size) if regular else None,
        }
    return {
        "files": files,
        "file_count": len(files),
        "bytes_hashed": sum(
            int(row["bytes"] or 0) for row in files.values()
        ),
        "passes": all(row["matches"] for row in files.values()),
    }


def dependency_audit(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    report = _hash_contract(EXPECTED_DEPENDENCY_SHA256, repo_root=repo_root)
    report["contract_sha256"] = canonical_json_hash(
        EXPECTED_DEPENDENCY_SHA256
    )
    report["source_or_artifact_payloads_parsed"] = False
    return report


def protected_source_audit(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    core = _hash_contract(CORE_GOVERNANCE_SHA256, repo_root=repo_root)
    roots = set(PROTECTED_DISCOVERY_ROOTS) | {"threes_rl/runs/dashboard"}
    root_checks = {}
    for relative in sorted(roots):
        path = repo_root / relative
        resolved = path.resolve()
        try:
            resolved.relative_to(repo_root.resolve())
            inside_repo = True
        except ValueError:
            inside_repo = False
        root_checks[relative] = {
            "exists": path.exists(),
            "directory": path.is_dir(),
            "nonsymlink": not path.is_symlink(),
            "inside_repo": inside_repo,
        }
    forbidden_checks = {}
    for relative in FORBIDDEN_BODY_PREFIXES:
        path = repo_root / relative
        resolved = path.resolve()
        try:
            resolved.relative_to(repo_root.resolve())
            inside_repo = True
        except ValueError:
            inside_repo = False
        forbidden_checks[relative] = {
            "exists": path.exists(),
            "inside_repo": inside_repo,
            "parse_forbidden": True,
        }
    live_checks = {}
    for relative in LIVE_SUMMARY_PATHS:
        path = repo_root / relative
        live_checks[relative] = {
            "exists": path.exists(),
            "regular_nonsymlink": (
                path.exists() and path.is_file() and not path.is_symlink()
            ),
            "byte_stability_binding_deferred": True,
            "parse_deferred": True,
        }
    checks = {
        "core_governance_hashes_exact": core["passes"],
        "discovery_roots_valid": all(
            row["exists"]
            and row["directory"]
            and row["nonsymlink"]
            and row["inside_repo"]
            for row in root_checks.values()
        ),
        "forbidden_prefixes_inside_repo": all(
            row["inside_repo"] for row in forbidden_checks.values()
        ),
        "live_summaries_classified": all(
            row["exists"] and row["regular_nonsymlink"]
            for row in live_checks.values()
        ),
        "governance_tokens_nonempty": bool(GOVERNANCE_FILENAME_TOKENS),
        "identity_fields_nonempty": bool(PROTECTED_IDENTITY_FIELDS),
        "all_three_identity_formats_frozen": (
            PROTECTED_DISCOVERY_EXTENSIONS == (".json", ".jsonl", ".csv")
        ),
        "hash_only_bodies_require_identity_companion": (
            HASH_ONLY_IDENTITY_COMPANION_REQUIRED
        ),
        "no_content_inventory_executed": True,
        "no_json_payload_parsed": True,
    }
    return {
        "core_governance": core,
        "discovery_patterns": PROTECTED_DISCOVERY_PATTERNS,
        "discovery_roots": root_checks,
        "governance_filename_tokens": GOVERNANCE_FILENAME_TOKENS,
        "protected_identity_fields": PROTECTED_IDENTITY_FIELDS,
        "forbidden_body_prefixes": forbidden_checks,
        "live_summaries": live_checks,
        "contract_sha256": PROTECTED_CONTRACT_SHA256,
        "content_inventory_executed": False,
        "json_payloads_parsed": False,
        "checks": checks,
        "passes": all(checks.values()),
    }


def family_contract_audit() -> dict[str, Any]:
    signatures = [FAMILY_SIGNATURES[name] for name in FAMILY_ORDER]
    checks = {
        "exact_four_families": len(FAMILY_ORDER) == MINIMUM_FAMILIES,
        "semantic_order_exact": tuple(FAMILY_SIGNATURES) == FAMILY_ORDER,
        "signatures_unique": len(set(signatures)) == len(signatures),
        "signatures_sha256_shaped": all(
            len(value) == 64
            and set(value).issubset(set("0123456789abcdef"))
            for value in signatures
        ),
        "policy_lock_sha256_shaped": (
            len(ACCEPTED_POLICY_LOCK_SHA256) == 64
            and set(ACCEPTED_POLICY_LOCK_SHA256).issubset(
                set("0123456789abcdef")
            )
        ),
        "future_family_cap_25_percent": 1.0 / len(FAMILY_ORDER)
        < MAX_FAMILY_SHARE,
    }
    return {
        "family_order": FAMILY_ORDER,
        "signatures": FAMILY_SIGNATURES,
        "accepted_policy_lock_sha256": ACCEPTED_POLICY_LOCK_SHA256,
        "checks": checks,
        "passes": all(checks.values()),
    }


def family_target_matrix(total: int) -> tuple[tuple[int, ...], ...]:
    if total <= 0 or total % len(FAMILY_ORDER):
        raise O6ContractError("Role total must be positive and divisible by 4")
    base, remainder = divmod(total, len(FAMILY_ORDER) * len(TARGET_ORDER))
    extras_by_remainder = {
        0: (),
        4: ((0, 0), (1, 1), (2, 2), (3, 0)),
        8: tuple(
            (family_index, target_index)
            for family_index in range(len(FAMILY_ORDER))
            for target_index in range(len(TARGET_ORDER))
            if (family_index, target_index)
            not in ((0, 2), (1, 0), (2, 1), (3, 2))
        ),
    }
    if remainder not in extras_by_remainder:
        raise O6ContractError(
            f"Frozen near-balance has no remainder rule for {remainder}"
        )
    matrix = [
        [base for _ in TARGET_ORDER]
        for _ in FAMILY_ORDER
    ]
    for family_index, target_index in extras_by_remainder[remainder]:
        matrix[family_index][target_index] += 1
    frozen = tuple(tuple(row) for row in matrix)
    row_totals = tuple(sum(row) for row in frozen)
    column_totals = tuple(
        sum(frozen[row][column] for row in range(len(FAMILY_ORDER)))
        for column in range(len(TARGET_ORDER))
    )
    if len(set(row_totals)) != 1:
        raise O6ContractError("Family rows are not exactly balanced")
    if max(column_totals) - min(column_totals) > 1:
        raise O6ContractError("Target columns differ by more than one")
    if sum(row_totals) != total:
        raise O6ContractError("Matrix total mismatch")
    return frozen


def role_matrix_contract(untouched_n: int) -> dict[str, Any]:
    if untouched_n not in ROLE_COUNTS_BY_UNTOUCHED_N:
        raise O6ContractError(f"Unsupported untouched N: {untouched_n}")
    counts = ROLE_COUNTS_BY_UNTOUCHED_N[untouched_n]
    matrices = {
        role: family_target_matrix(counts[role]) for role in ROLE_ORDER
    }
    family_counts = {
        role: tuple(sum(row) for row in matrices[role])
        for role in ROLE_ORDER
    }
    target_counts = {
        role: tuple(
            sum(matrices[role][row][column] for row in range(4))
            for column in range(3)
        )
        for role in ROLE_ORDER
    }
    checks = {
        "roles_exact": tuple(matrices) == ROLE_ORDER,
        "counts_exact": all(
            sum(sum(row) for row in matrices[role]) == counts[role]
            for role in ROLE_ORDER
        ),
        "families_exactly_balanced": all(
            len(set(family_counts[role])) == 1 for role in ROLE_ORDER
        ),
        "targets_near_balanced": all(
            max(target_counts[role]) - min(target_counts[role]) <= 1
            for role in ROLE_ORDER
        ),
        "family_cap_below_40_percent": all(
            max(family_counts[role]) / counts[role] < MAX_FAMILY_SHARE
            for role in ROLE_ORDER
        ),
    }
    return {
        "untouched_n": untouched_n,
        "role_counts": counts,
        "matrices": matrices,
        "family_counts": family_counts,
        "target_counts": target_counts,
        "checks": checks,
        "passes": all(checks.values()),
    }


def candidate_selection_key(row: Mapping[str, Any]) -> str:
    ancestry = str(row["ancestry"])
    state_hash = str(row["state_hash"])
    frame_index = int(row["frame_index"])
    target = int(row["target"])
    raw_pair = row["pair_coords"]
    if (
        not isinstance(raw_pair, (list, tuple))
        or len(raw_pair) != 2
    ):
        raise O6ContractError("pair_coords must contain exactly two cells")
    pair = tuple(
        tuple(int(value) for value in coordinate) for coordinate in raw_pair
    )
    if any(len(coordinate) != 2 for coordinate in pair):
        raise O6ContractError("Each pair coordinate must have two dimensions")
    if target not in TARGET_ORDER:
        raise O6ContractError(f"Unsupported target: {target}")
    material = (
        f"O6-P0-root-v1|{ancestry}|{state_hash}|{frame_index}|"
        f"{target}|{pair}"
    )
    return hashlib.sha256(material.encode("ascii")).hexdigest()


def dedupe_one_candidate_per_ancestry(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        ancestry = str(row["ancestry"])
        family = str(row["family"])
        if family not in FAMILY_ORDER:
            raise O6ContractError(f"Unsupported family: {family}")
        row["target"] = int(row["target"])
        row["selection_key"] = candidate_selection_key(row)
        prior = chosen.get(ancestry)
        if prior is None or row["selection_key"] < prior["selection_key"]:
            chosen[ancestry] = row
    return sorted(chosen.values(), key=lambda row: row["selection_key"])


def allocate_roles_without_backtracking(
    rows: Sequence[Mapping[str, Any]],
    *,
    untouched_n: int,
) -> dict[str, Any]:
    deduped = dedupe_one_candidate_per_ancestry(rows)
    matrix_contract = role_matrix_contract(untouched_n)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        cells[(str(row["family"]), int(row["target"]))].append(row)
    for candidates in cells.values():
        candidates.sort(key=lambda row: row["selection_key"])

    selected: list[dict[str, Any]] = []
    deficits: list[dict[str, Any]] = []
    offsets: Counter[tuple[str, int]] = Counter()
    for role in ROLE_ORDER:
        matrix = matrix_contract["matrices"][role]
        for family_index, family in enumerate(FAMILY_ORDER):
            for target_index, target in enumerate(TARGET_ORDER):
                required = int(matrix[family_index][target_index])
                key = (family, target)
                start = offsets[key]
                stop = start + required
                available = cells.get(key, [])
                picked = available[start:stop]
                if len(picked) != required:
                    deficits.append(
                        {
                            "role": role,
                            "family": family,
                            "target": target,
                            "required": required,
                            "available_remaining": max(0, len(available) - start),
                        }
                    )
                selected.extend({**row, "role": role} for row in picked)
                offsets[key] = stop
    ancestries = [str(row["ancestry"]) for row in selected]
    role_counts = Counter(str(row["role"]) for row in selected)
    checks = {
        "no_deficits": not deficits,
        "ancestries_unique": len(ancestries) == len(set(ancestries)),
        "role_counts_exact": all(
            role_counts[role]
            == ROLE_COUNTS_BY_UNTOUCHED_N[untouched_n][role]
            for role in ROLE_ORDER
        ),
        "one_root_per_ancestry": len(ancestries) == len(selected),
    }
    return {
        "selected": selected,
        "deficits": deficits,
        "deduped_candidate_count": len(deduped),
        "selected_count": len(selected),
        "selection_sha256": canonical_json_hash(selected),
        "matrix_contract": matrix_contract,
        "checks": checks,
        "passes": all(checks.values()),
    }


def partition_integrity(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_role_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    ancestries = [str(row["ancestry"]) for row in normalized]
    role_counts = Counter(str(row["role"]) for row in normalized)
    family_role_counts: dict[str, Counter[str]] = {
        role: Counter(
            str(row["family"])
            for row in normalized
            if str(row["role"]) == role
        )
        for role in ROLE_ORDER
    }
    duplicate_ancestries = sorted(
        ancestry
        for ancestry, count in Counter(ancestries).items()
        if count > 1
    )
    unknown_roles = sorted(set(role_counts) - set(ROLE_ORDER))
    unknown_families = sorted(
        {
            str(row["family"])
            for row in normalized
            if str(row["family"]) not in FAMILY_ORDER
        }
    )
    family_shares = {
        role: {
            family: (
                family_role_counts[role][family] / role_counts[role]
                if role_counts[role]
                else 0.0
            )
            for family in FAMILY_ORDER
        }
        for role in ROLE_ORDER
    }
    checks = {
        "ancestry_unique_across_roles": not duplicate_ancestries,
        "roles_known": not unknown_roles,
        "families_known": not unknown_families,
        "at_least_four_families_overall": (
            len({str(row["family"]) for row in normalized}) >= MINIMUM_FAMILIES
        ),
        "family_cap_below_40_percent_each_role": all(
            not role_counts[role]
            or max(family_shares[role].values()) < MAX_FAMILY_SHARE
            for role in ROLE_ORDER
        ),
        "expected_role_counts_exact_or_deferred": (
            expected_role_counts is None
            or all(
                role_counts[role] == int(expected_role_counts[role])
                for role in ROLE_ORDER
            )
        ),
    }
    return {
        "row_count": len(normalized),
        "role_counts": dict(role_counts),
        "family_role_counts": {
            role: dict(counts) for role, counts in family_role_counts.items()
        },
        "family_shares": family_shares,
        "duplicate_ancestries": duplicate_ancestries,
        "unknown_roles": unknown_roles,
        "unknown_families": unknown_families,
        "checks": checks,
        "passes": all(checks.values()),
    }


def _json_round_trip(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def whitelisted_current_state_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    board = payload["board"]
    preview = payload["preview"]
    tile_cycle = payload["tile_cycle"]
    if not isinstance(tile_cycle, Mapping):
        raise O6ContractError("tile_cycle must be a mapping")
    extracted = {
        "board": board,
        "preview": preview,
        "tile_cycle": {
            field: tile_cycle[field]
            for field in SOURCE_STATE_SCHEMA["tile_cycle_fields"]
        },
        "move_count": payload["move_count"],
        "game_over": payload["game_over"],
    }
    normalized = _json_round_trip(extracted)
    board_rows = normalized["board"]
    if (
        not isinstance(board_rows, list)
        or len(board_rows) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in board_rows)
    ):
        raise O6ContractError("board must be a 4x4 array")
    flat = [value for row in board_rows for value in row]
    if not all(isinstance(value, int) and value >= 0 for value in flat):
        raise O6ContractError("board values must be nonnegative integers")
    if not isinstance(normalized["move_count"], int):
        raise O6ContractError("move_count must be an integer")
    if not isinstance(normalized["game_over"], bool):
        raise O6ContractError("game_over must be boolean")
    return normalized


def current_state_round_trip(payload: Mapping[str, Any]) -> dict[str, Any]:
    first = whitelisted_current_state_payload(payload)
    second = whitelisted_current_state_payload(_json_round_trip(first))
    return {
        "payload": second,
        "sha256": canonical_json_hash(second),
        "stable": first == second,
        "schema_sha256": SOURCE_STATE_SCHEMA_SHA256,
    }


def validate_normalized_values(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = bool(np.isfinite(array).all())
    bounded = bool(((array >= 0.0) & (array <= 1.0)).all()) if finite else False
    return {
        "count": int(array.size),
        "finite": finite,
        "bounded_0_1": bounded,
        "passes": bool(array.size and finite and bounded),
    }


def validate_competing_risk_fixture(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_sha256") != RISK_SCHEMA_SHA256:
        raise O6ContractError("Risk fixture schema hash mismatch")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or not 1 <= len(raw_rows) <= 40:
        raise O6ContractError("Risk fixture must contain 1..40 rows")
    statuses: list[str] = []
    absorbed: str | None = None
    for index, raw_row in enumerate(raw_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise O6ContractError("Risk fixture row must be a mapping")
        if absorbed is not None:
            raise O6ContractError("Rows after an absorbing event are forbidden")
        if int(raw_row.get("t", -1)) != index:
            raise O6ContractError("Risk fixture row order is invalid")
        time_fraction = float(raw_row.get("time_fraction", math.nan))
        if time_fraction != index / int(RISK_SCHEMA["horizon"]):
            raise O6ContractError("Risk fixture time fraction is invalid")
        indicators = (
            float(raw_row.get("safe_merge_event", math.nan)),
            float(raw_row.get("competing_failure_event", math.nan)),
            float(raw_row.get("live_after_transition", math.nan)),
        )
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (time_fraction, *indicators)
        ):
            raise O6ContractError("Risk row is nonfinite or outside [0,1]")
        if sum(indicators) != 1.0:
            raise O6ContractError("Risk indicators must be one-hot")
        matching = [
            status
            for status, expected in RISK_SCHEMA["row_mapping"].items()
            if tuple(float(value) for value in expected) == indicators
        ]
        if len(matching) != 1:
            raise O6ContractError("Risk row has no unique source status")
        status = matching[0]
        statuses.append(status)
        if status != "live":
            absorbed = status
    administrative_censor = absorbed is None and len(statuses) == 40
    if absorbed is None and not administrative_censor:
        raise O6ContractError("Short all-live trajectories are incomplete")
    success_band = None
    if absorbed == "success":
        event_time = len(statuses)
        success_band = next(
            index
            for index, (lower, upper) in enumerate(
                RISK_SCHEMA["success_time_bands"]
            )
            if lower <= event_time <= upper
        )
    expected_summary = {
        "absorbing_status": absorbed,
        "administrative_h40_censor": administrative_censor,
        "success_time_band": success_band,
    }
    for field, expected in expected_summary.items():
        if payload.get(field) != expected:
            raise O6ContractError(f"Risk fixture {field} mismatch")
    normalized = _json_round_trip(payload)
    return {
        "normalized": normalized,
        "statuses": statuses,
        **expected_summary,
        "fixture_sha256": canonical_json_hash(normalized),
        "passes": True,
    }


def competing_risk_round_trip(payload: Mapping[str, Any]) -> dict[str, Any]:
    first = validate_competing_risk_fixture(payload)
    serialized = json.dumps(
        first["normalized"],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    reloaded = json.loads(serialized)
    second = validate_competing_risk_fixture(reloaded)
    return {
        "fixture_sha256": first["fixture_sha256"],
        "stable": (
            first["normalized"] == second["normalized"]
            and first["fixture_sha256"] == second["fixture_sha256"]
        ),
    }


def beta_parameters(mean: float, rho: float) -> tuple[float, float]:
    if not 0.0 < mean < 1.0:
        raise O6ContractError("Beta mean must be in (0,1)")
    if not 0.0 < rho < 1.0:
        raise O6ContractError("Beta ICC must be in (0,1)")
    concentration = (1.0 / rho) - 1.0
    return mean * concentration, (1.0 - mean) * concentration


def odds_shift(probability: np.ndarray, odds_ratio: float) -> np.ndarray:
    if odds_ratio <= 0.0:
        raise O6ContractError("Odds ratio must be positive")
    probability = np.asarray(probability, dtype=np.float64)
    odds = probability / np.clip(1.0 - probability, 1e-15, None)
    shifted = odds_ratio * odds
    return shifted / (1.0 + shifted)


def power_seed(n: int, odds_ratio: float, rho: float) -> int:
    return (
        2026072906
        + 100_000 * int(n)
        + 1_000 * round(100 * float(odds_ratio))
        + 10 * round(100 * float(rho))
    )


def power_strata_for_roots(n: int) -> np.ndarray:
    if n <= 0:
        raise O6ContractError("Power root count must be positive")
    return np.arange(n, dtype=np.int64) % len(POWER_STRATA)


def common_odds_ratio(
    control: np.ndarray,
    treatment: np.ndarray,
    strata: np.ndarray,
) -> float:
    control = np.asarray(control, dtype=np.int8)
    treatment = np.asarray(treatment, dtype=np.int8)
    strata = np.asarray(strata, dtype=np.int64)
    if control.shape != treatment.shape or control.ndim != 2:
        raise O6ContractError("Control/treatment must be paired root x repeat")
    if strata.shape != (control.shape[0],):
        raise O6ContractError("Strata must have one entry per root")
    numerator = 0.0
    denominator = 0.0
    for stratum in sorted(set(int(value) for value in strata)):
        mask = strata == stratum
        a = float(treatment[mask].sum()) + 0.5
        b = float(treatment[mask].size - treatment[mask].sum()) + 0.5
        c = float(control[mask].sum()) + 0.5
        d = float(control[mask].size - control[mask].sum()) + 0.5
        total = a + b + c + d
        numerator += a * d / total
        denominator += b * c / total
    if denominator <= 0.0:
        raise O6ContractError("Common-OR denominator is nonpositive")
    value = numerator / denominator
    if not math.isfinite(value) or value <= 0.0:
        raise O6ContractError("Common OR is invalid")
    return value


def frozen_power_grid_spec() -> dict[str, Any]:
    rows = [
        {
            "n": n,
            "true_or": odds_ratio,
            "rho": rho,
            "seed": power_seed(n, odds_ratio, rho),
            "datasets": POWER_DATASETS,
            "whole_root_bootstraps_per_dataset": POWER_BOOTSTRAPS,
        }
        for n in POWER_ROOT_COUNTS
        for odds_ratio in POWER_OR_GRID
        for rho in POWER_RHO_GRID
    ]
    checks = {
        "candidate_n_exact": tuple(
            dict.fromkeys(row["n"] for row in rows)
        ) == POWER_ROOT_COUNTS,
        "or_grid_exact": tuple(
            dict.fromkeys(row["true_or"] for row in rows)
        ) == POWER_OR_GRID,
        "rho_grid_exact": tuple(
            dict.fromkeys(row["rho"] for row in rows)
        ) == POWER_RHO_GRID,
        "all_seeds_unique": len({row["seed"] for row in rows}) == len(rows),
        "datasets_exact": all(
            row["datasets"] == POWER_DATASETS for row in rows
        ),
        "bootstraps_exact": all(
            row["whole_root_bootstraps_per_dataset"] == POWER_BOOTSTRAPS
            for row in rows
        ),
    }
    return {
        "rows": rows,
        "row_count": len(rows),
        "contract_sha256": POWER_CONTRACT_SHA256,
        "checks": checks,
        "passes": all(checks.values()),
    }


def power_workload_estimate() -> dict[str, Any]:
    cells = (
        len(POWER_ROOT_COUNTS)
        * len(POWER_OR_GRID)
        * len(POWER_RHO_GRID)
    )
    datasets = cells * POWER_DATASETS
    bootstraps = datasets * POWER_BOOTSTRAPS
    root_draws = sum(
        n
        * len(POWER_OR_GRID)
        * len(POWER_RHO_GRID)
        * POWER_DATASETS
        * POWER_BOOTSTRAPS
        for n in POWER_ROOT_COUNTS
    )
    checks = {
        "sixty_cells": cells == 60,
        "datasets_exact": datasets == 60 * 4_096,
        "bootstraps_exact": bootstraps == 60 * 4_096 * 4_096,
        "dataset_batches_integral": (
            POWER_DATASETS % POWER_DATASET_BATCH_SIZE == 0
        ),
        "bootstrap_batches_integral": (
            POWER_BOOTSTRAPS % POWER_BOOTSTRAP_BATCH_SIZE == 0
        ),
        "no_execution_api": True,
        "approximation_forbidden": True,
        "analytic_ci_substitution_forbidden": True,
    }
    return {
        "cells": cells,
        "datasets": datasets,
        "whole_root_bootstraps": bootstraps,
        "whole_root_index_draws": root_draws,
        "dataset_batch_size": POWER_DATASET_BATCH_SIZE,
        "dataset_batches_per_cell": (
            POWER_DATASETS // POWER_DATASET_BATCH_SIZE
        ),
        "bootstrap_batch_size": POWER_BOOTSTRAP_BATCH_SIZE,
        "bootstrap_batches_per_dataset": (
            POWER_BOOTSTRAPS // POWER_BOOTSTRAP_BATCH_SIZE
        ),
        "power_executed": False,
        "checks": checks,
        "passes": all(checks.values()),
    }


def select_power_design(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_cell = {
        (int(row["n"]), float(row["true_or"])): min(
            float(item["full_pass_power"])
            for item in rows
            if int(item["n"]) == int(row["n"])
            and float(item["true_or"]) == float(row["true_or"])
        )
        for row in rows
    }
    selected_n = next(
        (
            n
            for n in POWER_ROOT_COUNTS
            if by_cell.get((n, 1.50), -math.inf) >= POWER_REQUIRED
        ),
        None,
    )
    mde = None
    if selected_n is not None:
        mde = next(
            (
                odds_ratio
                for odds_ratio in POWER_OR_GRID
                if by_cell.get((selected_n, odds_ratio), -math.inf)
                >= POWER_REQUIRED
            ),
            None,
        )
    return {
        "selected_n": selected_n,
        "mde_grid_or": mde,
        "ready": selected_n is not None and mde is not None,
    }


def stream_window_contract() -> dict[str, Any]:
    ranges = []
    for purpose, bases in STREAM_WINDOWS.items():
        if tuple(bases) != STREAM_FIELDS:
            raise O6ContractError(f"Stream field order drift for {purpose}")
        for field, start in bases.items():
            ranges.append(
                {
                    "purpose": purpose,
                    "field": field,
                    "start": int(start),
                    "stop_exclusive": int(start) + STREAM_WINDOW_SIZE,
                }
            )
    ranges_sorted = sorted(ranges, key=lambda row: row["start"])
    overlaps = [
        (left, right)
        for left, right in zip(ranges_sorted, ranges_sorted[1:])
        if left["stop_exclusive"] > right["start"]
    ]
    checks = {
        "four_purposes": len(STREAM_WINDOWS) == 4,
        "four_fields_each": all(
            tuple(bases) == STREAM_FIELDS for bases in STREAM_WINDOWS.values()
        ),
        "sixteen_ranges": len(ranges) == 16,
        "one_million_each": all(
            row["stop_exclusive"] - row["start"] == STREAM_WINDOW_SIZE
            for row in ranges
        ),
        "internally_disjoint": not overlaps,
        "not_reserved_or_consumed": True,
    }
    return {
        "windows": STREAM_WINDOWS,
        "ranges": ranges,
        "overlaps": overlaps,
        "manifest_written": False,
        "streams_reserved": False,
        "streams_consumed": False,
        "contract_sha256": canonical_json_hash(STREAM_WINDOWS),
        "checks": checks,
        "passes": all(checks.values()),
    }


def zero_work_audit(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    paths = {
        "output_dir": repo_root / OUTPUT_DIR,
        "test_evidence": repo_root / TEST_EVIDENCE_PATH,
    }
    absent = {name: not path.exists() for name, path in paths.items()}
    checks = {
        "preflight_output_absent": absent["output_dir"],
        "test_evidence_absent": absent["test_evidence"],
        "marker_absent": not (paths["output_dir"] / "O6_P0_OPENED.json").exists(),
        "selection_absent": not (
            paths["output_dir"] / "O6_P0_SELECTED_ROOTS.json"
        ).exists(),
        "stream_manifest_absent": not (
            paths["output_dir"] / "O6_P0_STREAM_MANIFEST.json"
        ).exists(),
        "power_result_absent": not (
            paths["output_dir"] / "O6_P0_POWER_TABLE.json"
        ).exists(),
        "labels_models_outcomes_zero": True,
    }
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "absent": absent,
        "checks": checks,
        "passes": all(checks.values()),
    }


def preparation_audit(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    dependencies = dependency_audit(repo_root=repo_root)
    protected = protected_source_audit(repo_root=repo_root)
    families = family_contract_audit()
    matrices = {
        n: role_matrix_contract(n) for n in POWER_ROOT_COUNTS
    }
    power = frozen_power_grid_spec()
    power_workload = power_workload_estimate()
    streams = stream_window_contract()
    zero_work = zero_work_audit(repo_root=repo_root)
    source_files = {
        "charter": repo_root / CHARTER_PATH,
        "runner": repo_root / RUNNER_PATH,
        "tests": repo_root / TEST_PATH,
    }
    identities = {
        name: {
            "path": str(path),
            "sha256": sha256_path(path) if path.is_file() else None,
            "exists": path.is_file(),
        }
        for name, path in source_files.items()
    }
    checks = {
        "source_files_exist": all(row["exists"] for row in identities.values()),
        "dependencies_exact": dependencies["passes"],
        "protected_core_exact": protected["passes"],
        "families_exact": families["passes"],
        "all_candidate_role_matrices_exact": all(
            report["passes"] for report in matrices.values()
        ),
        "power_contract_frozen_not_executed": power["passes"],
        "power_workload_exact_not_executed": power_workload["passes"],
        "stream_windows_frozen_not_reserved": streams["passes"],
        "zero_work": zero_work["passes"],
        "source_scan_not_executed": True,
        "root_selection_not_executed": True,
        "historical_collision_scan_not_executed": True,
        "power_simulation_not_executed": True,
        "risk_schema_hash_exact": (
            RISK_SCHEMA_SHA256 == EXPECTED_RISK_SCHEMA_SHA256
        ),
        "source_state_schema_hash_exact": (
            SOURCE_STATE_SCHEMA_SHA256
            == EXPECTED_SOURCE_STATE_SCHEMA_SHA256
        ),
        "protected_contract_hash_exact": (
            PROTECTED_CONTRACT_SHA256
            == EXPECTED_PROTECTED_CONTRACT_SHA256
        ),
        "power_contract_hash_exact": (
            POWER_CONTRACT_SHA256 == EXPECTED_POWER_CONTRACT_SHA256
        ),
    }
    return {
        "version": VERSION,
        "source_identities": identities,
        "risk_schema_sha256": RISK_SCHEMA_SHA256,
        "source_state_schema_sha256": SOURCE_STATE_SCHEMA_SHA256,
        "dependency_audit": dependencies,
        "protected_source_audit": protected,
        "family_contract": families,
        "role_matrix_contracts": matrices,
        "power_contract": power,
        "power_workload_estimate": power_workload,
        "stream_window_contract": streams,
        "zero_work": zero_work,
        "checks": checks,
        "passes": all(checks.values()),
        "decision": (
            "CONTINUE_O6_SOURCE_PREPARATION_REVIEW"
            if all(checks.values())
            else "HOLD_O6_SOURCE_PREPARATION_INTEGRITY"
        ),
        "forbidden_work": {
            "markers": 0,
            "source_scans": 0,
            "root_selections": 0,
            "stream_reservations": 0,
            "stream_consumption": 0,
            "power_datasets": 0,
            "labels": 0,
            "training_steps": 0,
            "checkpoints": 0,
            "development_or_untouched_reads": 0,
            "policy_outcomes": 0,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only O6 source-preparation audit"
    )
    parser.add_argument(
        "command",
        choices=("audit-preparation",),
        help="Print the read-only preparation audit; writes nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "audit-preparation":
        raise O6ContractError(f"Unsupported command: {args.command}")
    print(json.dumps(preparation_audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Capture restart-program preflight state and safely prune rejected table payloads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "threes_rl" / "runs"
DEFAULT_OUT = RUNS / "forensics" / "restart_preflight" / "r0_preflight_20260709"


@dataclass(frozen=True)
class PruneCandidate:
    run_name: str
    reason: str
    evidence: str


PRUNE_CANDIDATES = (
    PruneCandidate(
        "td_phase4_promoted_balanced_restart_r1_v2_20260709",
        "Original R1 was harm-stopped and killed; its learned tables are permanently non-authoritative.",
        "ARTIFACT_RETENTION.md sections 'R1 Harm-Stop Retention' and 'R1b Confirmation Retention'; compact attribution evidence retained.",
    ),
    PruneCandidate(
        "td_phase4_incumbent_residual_r1b_v1_20260709",
        "R1b failed sealed confirmation, was permanently unpromoted, and may not be reused.",
        "ARTIFACT_RETENTION.md section 'R1b Confirmation Retention'; confirmation, metrics, audits, summaries, and replays retained.",
    ),
    PruneCandidate(
        "action_label_default_phase4_basecenter_confreg_swing13_endgame8_medium12_e80_a001_tc_20260706",
        "Base-centered medium-risk value sidecar explicitly recorded as non-promoted.",
        "EXPERIMENT_LOG.md lines 2470-2580; config/progress/meta and continuation conclusions retained.",
    ),
    PruneCandidate(
        "action_label_default_phase4_swing13_endgame8_highrisk16_e80_a001_tc_20260706",
        "Expanded high-risk correction failed its continuation gate and was killed.",
        "EXPERIMENT_LOG.md lines 2280-2450; config/progress/meta and continuation conclusions retained.",
    ),
    PruneCandidate(
        "action_label_default_phase4_weighted_confreg_swing13_endgame8_highrisk16_e80_a001_tc_20260706",
        "Confidence/regret high-risk correction failed its continuation gate and was not promoted.",
        "EXPERIMENT_LOG.md lines 2280-2450; config/progress/meta and continuation conclusions retained.",
    ),
    PruneCandidate(
        "action_label_default_phase4_weighted_confreg_swing13_endgame8_medium12_e80_a001_tc_20260706",
        "Weighted medium-risk value sidecar failed both ungated and risk-gated continuation screens.",
        "EXPERIMENT_LOG.md lines 2470-2580; config/progress/meta and continuation conclusions retained.",
    ),
    PruneCandidate(
        "action_label_default_phase4corner3_mediumonly_confreg_swing13_endgame8_medium12_e80_a001_tc_20260706",
        "Twelve-stage medium-risk sidecar was explicitly not promoted and worsened the incumbent continuation mean.",
        "EXPERIMENT_LOG.md lines 2710-2800; config/progress/meta and continuation conclusions retained.",
    ),
    PruneCandidate(
        "td_phase4corner3_parentinit_currentbest_reservoir_nstep_tc_80_20260707",
        "Historical 80-game phase4_corner3 model failed its normal-start gate by 14,172 score points.",
        "EXPERIMENT_LOG.md lines 3188-3278; config/summary/metrics/top-game replays/meta retained.",
    ),
    PruneCandidate(
        "transition_reachability_value_rawadj1536_extreme_role_no114_e120_a001_tc_20260707",
        "Exact-rung reachability sidecar was killed after reducing paired score and conversions.",
        "EXPERIMENT_LOG.md lines 14090-14370; config/summary/meta and paired-gate evidence retained.",
    ),
)


def run_output(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=False, capture_output=True, text=True)
    output = result.stdout.strip()
    if result.stderr.strip():
        output = f"{output}\n{result.stderr.strip()}".strip()
    return output


def directory_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def table_payloads(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.rglob("*.npy") if path.is_file())


def incumbent_paths() -> list[str]:
    policy_file = ROOT / "threes_rl" / "current_incumbent_policy.txt"
    lines = [line.strip() for line in policy_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        return []
    return [part for part in lines[-1].split(":") if part.startswith("threes_rl/runs/")]


def protected_top_replays() -> list[str]:
    audit_path = RUNS / "dashboard" / "replay_retention_audit.json"
    if not audit_path.exists():
        return []
    payload = json.loads(audit_path.read_text())
    paths: list[str] = []
    for row in payload.get("protected_global_top_replays", []):
        run_path = str(row.get("run_path", ""))
        for key in ("json", "html"):
            relative = str(row.get(key, ""))
            if relative:
                paths.append(str(RUNS / relative))
        if run_path:
            paths.append(str(RUNS / run_path))
    return sorted(set(paths))


def active_process_lines() -> list[str]:
    output = run_output(["ps", "-axo", "pid,etime,%cpu,%mem,command"])
    markers = ("threes_rl", "train_td", "dashboard")
    return [line for line in output.splitlines() if any(marker in line for marker in markers)]


def largest_run_children(limit: int = 20) -> list[dict[str, object]]:
    rows = [
        {"path": str(path.relative_to(ROOT)), "bytes": directory_bytes(path)}
        for path in RUNS.iterdir()
        if path.is_dir()
    ]
    return sorted(rows, key=lambda row: int(row["bytes"]), reverse=True)[:limit]


def compact_evidence(run_dir: Path) -> list[str]:
    names = ("config.json", "summary.json", "metrics.csv", "progress.csv", "progress.html")
    evidence = [str((run_dir / name).relative_to(ROOT)) for name in names if (run_dir / name).exists()]
    evidence.extend(str(path.relative_to(ROOT)) for path in sorted(run_dir.glob("top_games/**/replay.json")))
    evidence.extend(str(path.relative_to(ROOT)) for path in sorted(run_dir.glob("**/meta.json")))
    return evidence


def build_manifest(active_lines: Iterable[str]) -> list[dict[str, object]]:
    incumbent = incumbent_paths()
    protected = protected_top_replays()
    active = list(active_lines)
    rows: list[dict[str, object]] = []
    for candidate in PRUNE_CANDIDATES:
        run_dir = RUNS / candidate.run_name
        payloads = table_payloads(run_dir)
        relative_run = str(run_dir.relative_to(ROOT))
        active_hits = [line for line in active if candidate.run_name in line]
        incumbent_hits = [path for path in incumbent if candidate.run_name in path]
        protected_hits = [path for path in protected if candidate.run_name in path]
        evidence = compact_evidence(run_dir)
        rows.append(
            {
                "path": f"{relative_run}/**/*.npy",
                "byte_size": sum(path.stat().st_size for path in payloads),
                "file_count": len(payloads),
                "reason": candidate.reason,
                "evidence_retained": candidate.evidence,
                "retained_evidence_paths": evidence,
                "active_process_check": "clear" if not active_hits else active_hits,
                "protected_reference_check": "clear"
                if not incumbent_hits and not protected_hits
                else {"incumbent": incumbent_hits, "protected_replays": protected_hits},
                "eligible": bool(payloads and evidence and not active_hits and not incumbent_hits and not protected_hits),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "path",
        "byte_size",
        "file_count",
        "reason",
        "evidence_retained",
        "active_process_check",
        "protected_reference_check",
        "eligible",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def apply_manifest(rows: list[dict[str, object]]) -> int:
    reclaimed = 0
    by_pattern = {row["path"]: row for row in rows}
    for candidate in PRUNE_CANDIDATES:
        run_dir = RUNS / candidate.run_name
        row = by_pattern[f"{run_dir.relative_to(ROOT)}/**/*.npy"]
        if not row["eligible"]:
            continue
        for payload in table_payloads(run_dir):
            reclaimed += payload.stat().st_size
            payload.unlink()
        for directory in sorted((path for path in run_dir.rglob("*") if path.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
    return reclaimed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--apply", action="store_true", help="Delete only eligible .npy payloads after writing the manifest.")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    active = active_process_lines()
    rows = build_manifest(active)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    before = {
        "git_status": run_output(["git", "status", "--short", "--branch"]),
        "active_processes": active,
        "df_h": run_output(["df", "-h", str(ROOT)]),
        "runs_bytes": directory_bytes(RUNS),
        "largest_run_children": largest_run_children(),
        "incumbent_component_paths": incumbent_paths(),
        "parent_mc1000_path": "threes_rl/runs/td_default_corner2_mc_1000_init3000_a0005_20260706/latest",
        "protected_top_replay_paths": protected_top_replays(),
        "r1_source_catalogs": [
            "threes_rl/runs/forensics/phase0_replay_coverage_inventory/phase0_prefirst1536_retained_replay_inventory_20260709/phase0_replay_coverage_inventory.json",
            "threes_rl/runs/eval_artifacts/**/replay.json",
            "threes_rl/runs/replays/**/*.json",
            "threes_rl/runs/*/top_games/**/replay.json",
        ],
    }
    payload = {
        "created_at": created_at,
        "mode": "apply" if args.apply else "dry_run",
        "preflight": before,
        "deletion_manifest": rows,
        "eligible_bytes": sum(int(row["byte_size"]) for row in rows if row["eligible"]),
        "eligible_rows": sum(bool(row["eligible"]) for row in rows),
    }
    (out_dir / "preflight_and_deletion_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_csv(out_dir / "deletion_manifest.csv", rows)

    if args.apply:
        reclaimed = apply_manifest(rows)
        payload["bytes_reclaimed"] = reclaimed
        payload["post_cleanup"] = {
            "df_h": run_output(["df", "-h", str(ROOT)]),
            "runs_bytes": directory_bytes(RUNS),
            "largest_run_children": largest_run_children(),
        }
        (out_dir / "preflight_and_deletion_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(json.dumps({
        "mode": payload["mode"],
        "eligible_rows": payload["eligible_rows"],
        "eligible_bytes": payload["eligible_bytes"],
        "bytes_reclaimed": payload.get("bytes_reclaimed", 0),
        "artifact": str((out_dir / "preflight_and_deletion_manifest.json").relative_to(ROOT)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

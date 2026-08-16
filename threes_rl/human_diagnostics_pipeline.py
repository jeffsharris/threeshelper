"""Build replay, reservoir, and scan artifacts from observed human games."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from html import escape
from pathlib import Path
from typing import Any

from threes_rl.high_board_reservoir import run_from_args as run_reservoir
from threes_rl.import_human_replay import import_events_file, parse_starter
from threes_rl.replay_policy_agreement import parse_corner_risk_filters, parse_phase_filters, run as run_agreement
from threes_rl.run_artifacts import safe_name, write_json
from threes_rl.support_ladder_window_reservoir import run_from_args as run_support_ladder_windows
from threes_rl.swing_label import run_swing_labeling
from threes_rl.transition_window_reservoir import run_from_args as run_transition_windows


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _replay_paths_from_import_manifest(manifest: dict[str, Any]) -> list[Path]:
    replays = manifest.get("replays", [])
    if not isinstance(replays, list):
        return []
    paths: list[Path] = []
    for item in replays:
        if not isinstance(item, dict) or item.get("json") is None:
            continue
        paths.append(Path(str(item["json"])))
    return paths


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def _label_command(*, policy: str, scan_json: str, out_dir: Path, workers: int) -> list[str]:
    return [
        ".venv/bin/python",
        "-m",
        "threes_rl.swing_label",
        "--base-policy",
        policy,
        "--samples-json",
        scan_json,
        "--label-repeats",
        "8",
        "--horizons",
        "32",
        "--workers",
        str(max(1, int(workers))),
        "--checkpoint-labels",
        "--out-dir",
        str(out_dir),
    ]


def read_policy_file(path: Path) -> str:
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            return text
    raise ValueError(f"No policy spec found in {path}")


def resolve_policy(args: argparse.Namespace) -> str | None:
    policy = args.policy.strip() if isinstance(args.policy, str) and args.policy.strip() else None
    policy_file = getattr(args, "policy_file", None)
    if policy is not None and policy_file is not None:
        raise ValueError("Use --policy or --policy-file, not both")
    if policy is not None:
        return policy
    if policy_file is not None:
        return read_policy_file(Path(policy_file))
    return None


def write_report_html(path: Path, payload: dict[str, Any]) -> None:
    imports = payload.get("imports", [])
    reservoir = payload.get("reservoir", {})
    transition_windows = payload.get("transition_windows", {})
    support_ladder = payload.get("support_ladder", {})
    scan = payload.get("scan", {})
    agreement = payload.get("agreement", {})
    next_steps = payload.get("next_steps", [])
    if not isinstance(imports, list):
        imports = []
    if not isinstance(reservoir, dict):
        reservoir = {}
    if not isinstance(transition_windows, dict):
        transition_windows = {}
    if not isinstance(support_ladder, dict):
        support_ladder = {}
    if not isinstance(scan, dict):
        scan = {}
    if not isinstance(agreement, dict):
        agreement = {}
    if not isinstance(next_steps, list):
        next_steps = []

    import_rows = []
    for item in imports:
        if not isinstance(item, dict):
            continue
        import_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('source_events', '')))}</td>"
            f"<td>{escape(str(item.get('games_seen', '')))}</td>"
            f"<td>{escape(str(item.get('games_imported', '')))}</td>"
            f"<td>{escape(str(item.get('games_skipped', '')))}</td>"
            "</tr>"
        )

    reservoir_summary = reservoir.get("summary", {}) if isinstance(reservoir, dict) else {}
    if not isinstance(reservoir_summary, dict):
        reservoir_summary = {}
    transition_summary = transition_windows.get("summary", {}) if isinstance(transition_windows, dict) else {}
    if not isinstance(transition_summary, dict):
        transition_summary = {}
    support_summary = support_ladder.get("summary", {}) if isinstance(support_ladder, dict) else {}
    if not isinstance(support_summary, dict):
        support_summary = {}
    scan_summary = scan.get("summary", {}) if isinstance(scan, dict) else {}
    if not isinstance(scan_summary, dict):
        scan_summary = {}
    scan_stats = scan.get("scan_stats", {}) if isinstance(scan, dict) else {}
    if not isinstance(scan_stats, dict):
        scan_stats = scan_summary.get("scan_stats", {})
    if not isinstance(scan_stats, dict):
        scan_stats = {}
    agreement_summary = agreement.get("summary", {}) if isinstance(agreement, dict) else {}
    if not isinstance(agreement_summary, dict):
        agreement_summary = {}

    next_rows = []
    for step in next_steps:
        if not isinstance(step, dict):
            continue
        command = step.get("command_text") or _command_text([str(part) for part in step.get("command", [])])
        next_rows.append(
            "<tr>"
            f"<td>{escape(str(step.get('name', '')))}</td>"
            f"<td><code>{escape(str(command))}</code></td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Human Diagnostics Pipeline</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101214; --panel: #171c20; --line: #344049; --ink: #edf3ee; --muted: #aab4ad; --gold: #e4bd4b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 42px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 18px 0 8px; font-size: 17px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 16px 0; }}
    .metric, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; color: var(--gold); font-size: 22px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    code {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--gold); }}
  </style>
</head>
<body>
  <main>
    <h1>Human Diagnostics Pipeline</h1>
    <p class="muted">Observed games imported into replay/reservoir/scan artifacts for late-game diagnostics.</p>
    <section class="grid">
      <div class="metric"><span class="muted">Imported Games</span><b>{escape(str(payload.get('games_imported', 0)))}</b></div>
      <div class="metric"><span class="muted">Reservoir Records</span><b>{escape(str(reservoir_summary.get('records', 0)))}</b></div>
      <div class="metric"><span class="muted">Transition Windows</span><b>{escape(str(transition_summary.get('records', 0)))}</b></div>
      <div class="metric"><span class="muted">Support Windows</span><b>{escape(str(support_summary.get('records', 0)))}</b></div>
      <div class="metric"><span class="muted">Scan Samples</span><b>{escape(str(scan_stats.get('accepted_samples', 0)))}</b></div>
      <div class="metric"><span class="muted">Recorded In Top 2</span><b>{float(agreement_summary.get('recorded_in_top_two_rate') or 0.0):.1%}</b></div>
      <div class="metric"><span class="muted">Action Match</span><b>{float(agreement_summary.get('action_match_rate') or 0.0):.1%}</b></div>
      <div class="metric"><span class="muted">High-Conf Misses</span><b>{escape(str(agreement_summary.get('high_confidence_misses', 0)))}</b></div>
    </section>
    <section class="panel">
      <h2>Imports</h2>
      <table><thead><tr><th>Source</th><th>Seen</th><th>Imported</th><th>Skipped</th></tr></thead><tbody>{''.join(import_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Next Steps</h2>
      <table><thead><tr><th>Name</th><th>Command</th></tr></thead><tbody>{''.join(next_rows)}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    events_paths = _flatten_paths(args.events_jsonl)
    if not events_paths:
        raise ValueError("No --events-jsonl paths were provided")

    out_dir = Path(args.out_dir)
    import_dir = out_dir / "imported_replays"
    reservoir_dir = out_dir / "reservoir"
    transition_dir = out_dir / "transition_windows"
    support_ladder_dir = out_dir / "support_ladder_windows"
    scan_dir = out_dir / "top_two_scan"
    starter_tile = parse_starter(args.starter)

    import_manifests: list[dict[str, Any]] = []
    replay_paths: list[Path] = []
    for events_path in events_paths:
        source_dir = import_dir / safe_name(f"{events_path.parent.name}_{events_path.stem}")
        manifest = import_events_file(
            events_path,
            source_dir,
            starter_tile=starter_tile,
            min_valid_moves=args.min_valid_moves,
            write_replay_html=not args.no_replay_html,
        )
        import_manifests.append(manifest)
        replay_paths.extend(_replay_paths_from_import_manifest(manifest))

    reservoir_payload: dict[str, Any] | None = None
    transition_payload: dict[str, Any] | None = None
    support_ladder_payload: dict[str, Any] | None = None
    scan_payload: dict[str, Any] | None = None
    agreement_payload: dict[str, Any] | None = None
    next_steps: list[dict[str, Any]] = []
    policy = resolve_policy(args)

    if replay_paths:
        if not bool(getattr(args, "no_transition_windows", False)):
            transition_payload = run_transition_windows(
                argparse.Namespace(
                    replay_json=[replay_paths],
                    replay_glob=[],
                    targets=args.transition_targets,
                    window_size=args.transition_window_size,
                    no_failures=args.transition_no_failures,
                    starter=args.starter,
                    max_records=args.transition_max_records,
                    out_dir=transition_dir,
                )
            )
        if not bool(getattr(args, "no_support_ladder", False)):
            support_ladder_payload = run_support_ladder_windows(
                argparse.Namespace(
                    replay_json=[replay_paths],
                    replay_glob=[],
                    exclude_path_substring=[],
                    targets=args.support_ladder_targets,
                    window_size=args.support_ladder_window_size,
                    no_failures=args.support_ladder_no_failures,
                    starter=args.starter,
                    max_records=args.support_ladder_max_records,
                    fresh_root_only=False,
                    root_origin="human",
                    out_dir=support_ladder_dir,
                )
            )
        reservoir_payload = run_reservoir(
            argparse.Namespace(
                replay_json=[replay_paths],
                replay_glob=[],
                min_tile=args.min_tile,
                phase_filter=args.phase_filter,
                corner_risk_filter=args.corner_risk_filter,
                default_starter=args.starter,
                first_per=args.reservoir_first_per,
                max_records=args.reservoir_max_records,
                max_per_stratum=args.reservoir_max_per_stratum,
                sort_by=args.reservoir_sort_by,
                out_dir=reservoir_dir,
            )
        )

        if policy:
            if not bool(getattr(args, "no_agreement_report", False)):
                agreement_phase_args = getattr(args, "agreement_phase_filter", None) or args.phase_filter
                agreement_corner_args = getattr(args, "agreement_corner_risk_filter", None) or args.corner_risk_filter
                agreement_phase_filter = parse_phase_filters(agreement_phase_args)
                agreement_corner_filter = parse_corner_risk_filters(agreement_corner_args)
                agreement_min_tile = (
                    int(args.agreement_min_tile)
                    if getattr(args, "agreement_min_tile", None) is not None
                    else int(args.min_tile)
                )
                agreement_payload = run_agreement(
                    policy_spec=policy,
                    replay_paths=replay_paths,
                    out_dir=out_dir / "policy_agreement",
                    min_tile=agreement_min_tile,
                    phase_filter=agreement_phase_filter,
                    corner_risk_filter=agreement_corner_filter,
                    default_starter_tile=starter_tile,
                    max_records=int(getattr(args, "agreement_max_records", 0)),
                    high_confidence_margin=float(getattr(args, "agreement_high_confidence_margin", 0.01)),
                )
            cache_path = args.action_value_cache or (out_dir / "action_value_cache.json")
            scan_payload = run_swing_labeling(
                argparse.Namespace(
                    base_policy=policy,
                    sample_mode=args.sample_mode,
                    comparison_policy=None,
                    samples_json=None,
                    replay_json=None,
                    replay_glob=[],
                    state_json=[[Path(reservoir_payload["records_json"])]],
                    state_glob=[],
                    seeds="0:0",
                    starter=args.starter,
                    max_moves=5000,
                    margin_threshold=args.margin_threshold,
                    max_samples=args.scan_max_samples,
                    max_per_stratum=args.scan_max_per_stratum,
                    max_replays=0,
                    replay_start_index=0,
                    max_state_records=args.max_state_records,
                    state_start_index=0,
                    action_value_cache=cache_path,
                    first_per=args.scan_first_per,
                    anchor_min_tile=args.anchor_min_tile,
                    min_top_value=args.min_top_value,
                    geometry_min_tile=args.geometry_min_tile,
                    geometry_min_delta=args.geometry_min_delta,
                    support_min_tile=768,
                    support_target_min_tile=3072,
                    support_mask_mode="masked",
                    support_min_delta=50.0,
                    replay_base_action="policy",
                    phase_filter=args.scan_phase_filter,
                    corner_risk_filter=args.scan_corner_risk_filter,
                    progress_every_seeds=0,
                    progress_every_replays=0,
                    progress_every_state_records=args.progress_every_state_records,
                    label_repeats=1,
                    horizons="32",
                    label_seed=args.label_seed,
                    stability_threshold=args.stability_threshold,
                    workers=1,
                    repeat_chunk_size=4,
                    progress_every_label_chunks=0,
                    checkpoint_labels=False,
                    label_progress_json=None,
                    no_label=True,
                    out_dir=scan_dir,
                )
            )
            label_out = out_dir / "labels_r8_h32"
            command = _label_command(
                policy=policy,
                scan_json=str(scan_payload["json"]),
                out_dir=label_out,
                workers=args.label_workers,
            )
            next_steps.append(
                {
                    "name": "cheap_label_pilot",
                    "command": command,
                    "command_text": _command_text(command),
                }
            )

    payload = {
        "version": 1,
        "kind": "threes_human_diagnostics_pipeline",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "events_jsonl": [str(path) for path in events_paths],
        "games_seen": sum(int(manifest.get("games_seen", 0)) for manifest in import_manifests),
        "games_imported": sum(int(manifest.get("games_imported", 0)) for manifest in import_manifests),
        "games_skipped": sum(int(manifest.get("games_skipped", 0)) for manifest in import_manifests),
        "policy": policy,
        "replay_json": [str(path) for path in replay_paths],
        "imports": import_manifests,
        "reservoir": reservoir_payload or {},
        "transition_windows": transition_payload or {},
        "support_ladder": support_ladder_payload or {},
        "agreement": agreement_payload or {},
        "scan": scan_payload or {},
        "next_steps": next_steps,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "human_diagnostics_manifest.json"
    html_path = out_dir / "human_diagnostics.html"
    payload["json"] = str(manifest_path)
    payload["html"] = str(html_path)
    write_json(manifest_path, payload)
    write_report_html(html_path, payload)
    write_json(manifest_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-jsonl", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/human_diagnostics/latest"))
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--min-valid-moves", type=int, default=1)
    parser.add_argument("--no-replay-html", action="store_true")
    parser.add_argument("--policy", help="Optional frozen actor policy for a no-label diagnostic scan.")
    parser.add_argument("--policy-file", type=Path, help="Read the frozen actor policy spec from the first non-comment line.")
    parser.add_argument("--min-tile", type=int, default=1536)
    parser.add_argument("--phase-filter", action="append", default=["late,endgame"])
    parser.add_argument("--corner-risk-filter", action="append", default=["medium,high"])
    parser.add_argument(
        "--reservoir-first-per",
        choices=["none", "replay", "replay-phase", "replay-stratum"],
        default="none",
    )
    parser.add_argument("--reservoir-max-records", type=int, default=0)
    parser.add_argument("--reservoir-max-per-stratum", type=int, default=0)
    parser.add_argument("--reservoir-sort-by", choices=["source", "score", "max_tile", "move"], default="max_tile")
    parser.add_argument("--no-transition-windows", action="store_true")
    parser.add_argument("--transition-targets", default="1536,3072,6144")
    parser.add_argument("--transition-window-size", type=int, default=40)
    parser.add_argument("--transition-max-records", type=int, default=0)
    parser.add_argument("--transition-no-failures", action="store_true")
    parser.add_argument("--no-support-ladder", action="store_true")
    parser.add_argument(
        "--support-ladder-targets",
        default=(
            "raw_duplicate_768,raw_adjacent_768,raw_one_1536,"
            "raw_duplicate_1536,raw_near_adjacent_1536,raw_adjacent_1536,second_3072"
        ),
    )
    parser.add_argument("--support-ladder-window-size", type=int, default=40)
    parser.add_argument("--support-ladder-max-records", type=int, default=0)
    parser.add_argument("--support-ladder-no-failures", action="store_true")
    parser.add_argument("--sample-mode", choices=["top-two", "anchor-risk", "geometry-risk"], default="top-two")
    parser.add_argument("--margin-threshold", type=float, default=0.002)
    parser.add_argument("--min-top-value", type=float, default=5000.0)
    parser.add_argument("--scan-max-samples", type=int, default=24)
    parser.add_argument("--scan-max-per-stratum", type=int, default=4)
    parser.add_argument(
        "--scan-first-per",
        choices=["seed", "seed-phase", "seed-stratum", "replay", "replay-phase", "replay-stratum", "none"],
        default="replay-stratum",
    )
    parser.add_argument("--max-state-records", type=int, default=0)
    parser.add_argument("--scan-phase-filter", action="append", default=["late,endgame"])
    parser.add_argument("--scan-corner-risk-filter", action="append", default=["medium,high"])
    parser.add_argument("--anchor-min-tile", type=int, default=1536)
    parser.add_argument("--geometry-min-tile", type=int, default=1536)
    parser.add_argument("--geometry-min-delta", type=float, default=1.0)
    parser.add_argument("--action-value-cache", type=Path)
    parser.add_argument("--agreement-min-tile", type=int)
    parser.add_argument("--agreement-phase-filter", action="append")
    parser.add_argument("--agreement-corner-risk-filter", action="append")
    parser.add_argument("--agreement-max-records", type=int, default=0)
    parser.add_argument("--agreement-high-confidence-margin", type=float, default=0.01)
    parser.add_argument("--no-agreement-report", action="store_true")
    parser.add_argument("--progress-every-state-records", type=int, default=0)
    parser.add_argument("--label-seed", type=int, default=20260706)
    parser.add_argument("--stability-threshold", type=float, default=0.70)
    parser.add_argument("--label-workers", type=int, default=1)
    args = parser.parse_args()

    payload = run_pipeline(args)
    print(
        json.dumps(
            {
                "games_imported": payload["games_imported"],
                "reservoir_records": (payload.get("reservoir", {}).get("summary", {}) or {}).get("records", 0),
                "transition_window_records": (payload.get("transition_windows", {}).get("summary", {}) or {}).get(
                    "records",
                    0,
                ),
                "support_ladder_records": (payload.get("support_ladder", {}).get("summary", {}) or {}).get(
                    "records",
                    0,
                ),
                "agreement_records": (payload.get("agreement", {}).get("summary", {}) or {}).get("records", 0),
                "agreement_action_match_rate": (payload.get("agreement", {}).get("summary", {}) or {}).get(
                    "action_match_rate",
                    0,
                ),
                "scan_samples": (payload.get("scan", {}).get("scan_stats", {}) or {}).get("accepted_samples", 0),
                "json": payload["json"],
                "html": payload["html"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

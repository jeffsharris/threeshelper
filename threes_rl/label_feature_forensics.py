"""Feature forensics for swing-label action decisions."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from threes_rl.action_prior import FEATURE_NAMES, action_features
from threes_rl.run_artifacts import write_json
from threes_rl.train_td import state_from_replay_payload


@dataclass
class FeatureCase:
    id: str
    source_json: str
    seed: int | None
    move_count: int | None
    stratum: str
    base_action: str
    comparison_action: str
    winner: str | None
    stable: bool
    same_winner_across_horizons: bool
    confidence: float
    regret: float
    category: str
    deltas: dict[str, float]
    preferred_action: str | None
    alternate_action: str | None


def _flatten_paths(path_groups: list[list[Path]] | None) -> list[Path]:
    if not path_groups:
        return []
    return [path for group in path_groups for path in group]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _label_winner(label: dict[str, Any]) -> str | None:
    winner = label.get("stable_winner") or label.get("oracle_winner")
    if winner is None or str(winner) == "tie":
        return None
    return str(winner)


def _actions_for_item(item: dict[str, Any], label: dict[str, Any]) -> list[str]:
    raw_actions = label.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raw_actions = item.get("top_two_actions")
    actions = [str(action) for action in raw_actions or [] if action is not None]
    seen: set[str] = set()
    out: list[str] = []
    for action in actions:
        if action not in seen:
            out.append(action)
            seen.add(action)
    return out


def _preferred_and_alternate(
    *,
    actions: list[str],
    winner: str | None,
    base_action: str,
) -> tuple[str | None, str | None]:
    if not actions:
        return None, None
    if winner is not None and winner in actions:
        preferred = winner
    elif base_action in actions:
        preferred = base_action
    else:
        preferred = actions[0]
    if preferred != base_action and base_action in actions:
        return preferred, base_action
    for action in actions:
        if action != preferred:
            return preferred, action
    return preferred, None


def _category(stable: bool, winner: str | None, base_action: str) -> str:
    if winner is None:
        return "tie_or_no_winner"
    if stable and winner != base_action:
        return "stable_flip"
    if stable and winner == base_action:
        return "stable_keep"
    if not stable and winner != base_action:
        return "unstable_flip"
    return "unstable_keep"


def case_from_label_item(item: dict[str, Any], *, source_json: Path, starter_tile: int | None) -> FeatureCase | None:
    if not isinstance(item.get("state"), dict) or not isinstance(item.get("label"), dict):
        return None
    label = item["label"]
    actions = _actions_for_item(item, label)
    if len(actions) < 2:
        return None
    base_action = str(item.get("base_action") or actions[0])
    winner = _label_winner(label)
    stable = bool(label.get("stable"))
    preferred, alternate = _preferred_and_alternate(
        actions=actions,
        winner=winner,
        base_action=base_action,
    )
    if preferred is None or alternate is None:
        return None
    state = state_from_replay_payload(item["state"])
    preferred_features = action_features(state, preferred, starter_tile=starter_tile)
    alternate_features = action_features(state, alternate, starter_tile=starter_tile)
    deltas = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, preferred_features - alternate_features)
    }
    features = item.get("features") if isinstance(item.get("features"), dict) else {}
    return FeatureCase(
        id=str(item.get("id", "")),
        source_json=str(source_json),
        seed=_safe_int(item.get("seed")),
        move_count=_safe_int(item.get("move_count")),
        stratum=str(features.get("stratum", "unknown")),
        base_action=base_action,
        comparison_action=str(item.get("comparison_action") or ""),
        winner=winner,
        stable=stable,
        same_winner_across_horizons=bool(label.get("same_winner_across_horizons")),
        confidence=_safe_float(label.get("min_bootstrap_winner_fraction"), 0.5),
        regret=_safe_float(label.get("oracle_regret_at_max_horizon"), 0.0),
        category=_category(stable, winner, base_action),
        deltas=deltas,
        preferred_action=preferred,
        alternate_action=alternate,
    )


def load_cases(paths: Iterable[Path], *, starter_tile: int | None = 1536) -> list[FeatureCase]:
    cases: list[FeatureCase] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            continue
        for item in labels:
            if not isinstance(item, dict):
                continue
            case = case_from_label_item(item, source_json=Path(path), starter_tile=starter_tile)
            if case is not None:
                cases.append(case)
    return cases


def _mean_delta(cases: list[FeatureCase], feature: str) -> float:
    values = [case.deltas.get(feature, 0.0) for case in cases]
    return float(mean(values)) if values else 0.0


def _top_feature_rows(cases: list[FeatureCase], *, limit: int) -> list[dict[str, float | str]]:
    rows = []
    for feature in FEATURE_NAMES:
        value = _mean_delta(cases, feature)
        rows.append({"feature": feature, "mean_delta": float(value), "abs_mean_delta": abs(float(value))})
    rows.sort(key=lambda row: (-float(row["abs_mean_delta"]), str(row["feature"])))
    return rows[: int(limit)]


def _contrast_rows(
    left: list[FeatureCase],
    right: list[FeatureCase],
    *,
    limit: int,
    include_action_features: bool = True,
) -> list[dict[str, float | str]]:
    rows = []
    for feature in FEATURE_NAMES:
        if not include_action_features and feature.startswith("action_"):
            continue
        left_mean = _mean_delta(left, feature)
        right_mean = _mean_delta(right, feature)
        contrast = left_mean - right_mean
        rows.append(
            {
                "feature": feature,
                "left_mean_delta": float(left_mean),
                "right_mean_delta": float(right_mean),
                "contrast": float(contrast),
                "abs_contrast": abs(float(contrast)),
            }
        )
    rows.sort(key=lambda row: (-float(row["abs_contrast"]), str(row["feature"])))
    return rows[: int(limit)]


def summarize_cases(cases: list[FeatureCase], *, top_features: int = 16) -> dict[str, Any]:
    by_category: dict[str, list[FeatureCase]] = {}
    for case in cases:
        by_category.setdefault(case.category, []).append(case)
    stable_flips = by_category.get("stable_flip", [])
    stable_keeps = by_category.get("stable_keep", [])
    positive_flips = [case for case in stable_flips if case.regret > 0.0]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cases": len(cases),
        "categories": dict(Counter(case.category for case in cases)),
        "strata": dict(Counter(case.stratum for case in cases)),
        "stable_flip_regret": {
            "count": len(positive_flips),
            "mean": float(mean(case.regret for case in positive_flips)) if positive_flips else 0.0,
            "max": float(max((case.regret for case in positive_flips), default=0.0)),
        },
        "top_stable_flip_feature_deltas": _top_feature_rows(stable_flips, limit=top_features),
        "top_stable_keep_feature_deltas": _top_feature_rows(stable_keeps, limit=top_features),
        "stable_flip_minus_keep_contrasts": _contrast_rows(stable_flips, stable_keeps, limit=top_features),
        "stable_flip_minus_keep_non_action_contrasts": _contrast_rows(
            stable_flips,
            stable_keeps,
            limit=top_features,
            include_action_features=False,
        ),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    cases = payload.get("cases", [])
    contrast = summary.get("stable_flip_minus_keep_contrasts", []) if isinstance(summary, dict) else []
    non_action_contrast = (
        summary.get("stable_flip_minus_keep_non_action_contrasts", [])
        if isinstance(summary, dict)
        else []
    )

    def cell(value: object) -> str:
        return escape(str(value))

    case_rows = []
    for case in cases[:200] if isinstance(cases, list) else []:
        case_rows.append(
            "<tr>"
            f"<td>{cell(case.get('category'))}</td>"
            f"<td>{cell(case.get('seed'))}</td>"
            f"<td>{cell(case.get('move_count'))}</td>"
            f"<td>{cell(case.get('stratum'))}</td>"
            f"<td>{cell(case.get('base_action'))}</td>"
            f"<td>{cell(case.get('winner'))}</td>"
            f"<td>{float(case.get('regret') or 0.0):.1f}</td>"
            f"<td>{float(case.get('confidence') or 0.0):.3f}</td>"
            "</tr>"
        )
    contrast_rows = []
    for row in contrast[:30] if isinstance(contrast, list) else []:
        contrast_rows.append(
            "<tr>"
            f"<td>{cell(row.get('feature'))}</td>"
            f"<td>{float(row.get('left_mean_delta') or 0.0):.4f}</td>"
            f"<td>{float(row.get('right_mean_delta') or 0.0):.4f}</td>"
            f"<td>{float(row.get('contrast') or 0.0):.4f}</td>"
            "</tr>"
        )
    non_action_rows = []
    for row in non_action_contrast[:30] if isinstance(non_action_contrast, list) else []:
        non_action_rows.append(
            "<tr>"
            f"<td>{cell(row.get('feature'))}</td>"
            f"<td>{float(row.get('left_mean_delta') or 0.0):.4f}</td>"
            f"<td>{float(row.get('right_mean_delta') or 0.0):.4f}</td>"
            f"<td>{float(row.get('contrast') or 0.0):.4f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Threes Label Feature Forensics</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101214; --panel: #171c20; --line: #344049; --ink: #edf3ee; --muted: #aab4ad; --gold: #e4bd4b; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 18px 0 8px; font-size: 17px; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin: 16px 0; }}
    .metric, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; color: var(--gold); font-size: 23px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(4), td:nth-child(4) {{ text-align: left; }}
  </style>
</head>
<body>
  <main>
    <h1>Label Feature Forensics</h1>
    <p class="muted">Preferred-action minus alternate-action deltas over action-conditioned features.</p>
    <section class="grid">
      <div class="metric"><span class="muted">Cases</span><b>{cell(summary.get('cases', 0) if isinstance(summary, dict) else 0)}</b></div>
      <div class="metric"><span class="muted">Stable Flips</span><b>{cell((summary.get('categories') or {}).get('stable_flip', 0) if isinstance(summary, dict) else 0)}</b></div>
      <div class="metric"><span class="muted">Stable Keeps</span><b>{cell((summary.get('categories') or {}).get('stable_keep', 0) if isinstance(summary, dict) else 0)}</b></div>
    </section>
    <section class="panel">
      <h2>Flip Minus Keep Feature Contrasts</h2>
      <table><thead><tr><th>Feature</th><th>Flip Mean</th><th>Keep Mean</th><th>Contrast</th></tr></thead><tbody>{''.join(contrast_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Non-Action Feature Contrasts</h2>
      <table><thead><tr><th>Feature</th><th>Flip Mean</th><th>Keep Mean</th><th>Contrast</th></tr></thead><tbody>{''.join(non_action_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top: 14px;">
      <h2>Cases</h2>
      <table><thead><tr><th>Category</th><th>Seed</th><th>Move</th><th>Stratum</th><th>Base</th><th>Winner</th><th>Regret</th><th>Confidence</th></tr></thead><tbody>{''.join(case_rows)}</tbody></table>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run(label_json: list[Path], out_dir: Path, *, starter_tile: int | None = 1536, top_features: int = 16) -> dict[str, Any]:
    cases = load_cases(label_json, starter_tile=starter_tile)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "label_json": [str(path) for path in label_json],
        "summary": summarize_cases(cases, top_features=top_features),
        "cases": [
            {
                "id": case.id,
                "source_json": case.source_json,
                "seed": case.seed,
                "move_count": case.move_count,
                "stratum": case.stratum,
                "base_action": case.base_action,
                "comparison_action": case.comparison_action,
                "winner": case.winner,
                "stable": case.stable,
                "same_winner_across_horizons": case.same_winner_across_horizons,
                "confidence": case.confidence,
                "regret": case.regret,
                "category": case.category,
                "preferred_action": case.preferred_action,
                "alternate_action": case.alternate_action,
                "deltas": case.deltas,
            }
            for case in cases
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "label_feature_forensics.json", payload)
    write_html(out_dir / "label_feature_forensics.html", payload)
    return payload


def parse_starter(text: str) -> int | None:
    value = text.strip().lower()
    return None if value == "none" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-json", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--top-features", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/label_features/latest"))
    args = parser.parse_args()
    paths = _flatten_paths(args.label_json)
    payload = run(
        paths,
        args.out_dir,
        starter_tile=parse_starter(args.starter),
        top_features=args.top_features,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={args.out_dir / 'label_feature_forensics.json'}")
    print(f"html={args.out_dir / 'label_feature_forensics.html'}")


if __name__ == "__main__":
    main()

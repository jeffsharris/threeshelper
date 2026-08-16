"""Summarize promotion-rate labels into action-sensitivity buckets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from threes_rl.run_artifacts import write_json


def _load_labels(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        raise ValueError(f"{path} does not contain labels[]")
    return [item for item in labels if isinstance(item, dict)]


def _max_horizon_row(label: dict[str, Any]) -> dict[str, Any]:
    rows = label.get("label", {}).get("horizon_results", [])
    if not isinstance(rows, list) or not rows:
        return {}
    return max((row for row in rows if isinstance(row, dict)), key=lambda row: int(row.get("horizon", 0)), default={})


def classify_label(item: dict[str, Any], *, rate_gap: float) -> dict[str, Any]:
    row = _max_horizon_row(item)
    rates = row.get("promotion_rate", {})
    if not isinstance(rates, dict) or not rates:
        bucket = "no_promotion_metric"
        min_rate = max_rate = rate_span = 0.0
    else:
        values = [float(value) for value in rates.values()]
        min_rate = min(values)
        max_rate = max(values)
        rate_span = max_rate - min_rate
        if min_rate >= 1.0:
            bucket = "inevitable"
        elif max_rate <= 0.0:
            bucket = "unreachable_by_horizon"
        elif rate_span >= float(rate_gap):
            bucket = "action_sensitive"
        else:
            bucket = "action_insensitive_mixed"
    label = item.get("label", {}) if isinstance(item.get("label"), dict) else {}
    return {
        "id": item.get("id"),
        "source_replay": item.get("source_replay"),
        "source_seed": item.get("source_seed", item.get("seed")),
        "source_frame_index": item.get("source_frame_index"),
        "move_count": item.get("move_count"),
        "moves_to_promotion": item.get("moves_to_promotion"),
        "stratum": (item.get("features") or {}).get("stratum") if isinstance(item.get("features"), dict) else None,
        "base_action": item.get("base_action"),
        "comparison_action": item.get("comparison_action"),
        "stable": bool(label.get("stable")),
        "stable_winner": label.get("stable_winner"),
        "oracle_regret": float(label.get("oracle_regret_at_max_horizon") or 0.0),
        "bucket": bucket,
        "max_horizon": int(row.get("horizon", 0)) if row else None,
        "promotion_rate": rates,
        "promotion_winner": row.get("promotion_winner"),
        "promotion_rate_min": float(min_rate),
        "promotion_rate_max": float(max_rate),
        "promotion_rate_span": float(rate_span),
        "promotion_rate_gain_vs_base": float(label.get("promotion_rate_gain_vs_base_at_max_horizon") or 0.0),
        "mean_delta": row.get("mean_delta", {}),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = Counter(str(row.get("bucket")) for row in rows)
    action_sensitive = [row for row in rows if row.get("bucket") == "action_sensitive"]
    gains = [float(row.get("promotion_rate_gain_vs_base") or 0.0) for row in rows]
    positive_gains = [gain for gain in gains if gain > 0.0]
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "labels": len(rows),
        "buckets": dict(buckets),
        "action_sensitive_labels": len(action_sensitive),
        "positive_promotion_gains": len(positive_gains),
        "mean_positive_promotion_gain": float(mean(positive_gains)) if positive_gains else 0.0,
        "max_positive_promotion_gain": float(max(positive_gains)) if positive_gains else 0.0,
        "stable_action_sensitive": sum(1 for row in action_sensitive if bool(row.get("stable"))),
    }


def write_html(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])

    def cell(value: object) -> str:
        return escape(str(value))

    table_rows = []
    for row in rows if isinstance(rows, list) else []:
        table_rows.append(
            "<tr>"
            f"<td>{cell(row.get('bucket'))}</td>"
            f"<td>{cell(row.get('source_seed'))}</td>"
            f"<td>{cell(row.get('move_count'))}</td>"
            f"<td>{cell(row.get('moves_to_promotion'))}</td>"
            f"<td>{cell(row.get('base_action'))}/{cell(row.get('comparison_action'))}</td>"
            f"<td>{cell(row.get('promotion_rate'))}</td>"
            f"<td>{float(row.get('promotion_rate_span') or 0.0):.3f}</td>"
            f"<td>{float(row.get('promotion_rate_gain_vs_base') or 0.0):.3f}</td>"
            f"<td>{float(row.get('oracle_regret') or 0.0):.0f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Promotion Label Report</title>
  <style>
    :root {{ color-scheme: dark; --bg: #101318; --panel: #171d24; --line: #34404d; --text: #edf2f7; --muted: #aab6c2; --gold: #f2c14e; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; padding: 24px 0 40px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card, .panel {{ border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ margin-top: 4px; color: var(--gold); font-size: 22px; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: right; vertical-align: top; }}
    th:first-child, td:first-child, th:nth-child(5), td:nth-child(5), th:nth-child(6), td:nth-child(6) {{ text-align: left; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>Promotion Label Report</h1>
    <p class="muted">Buckets top-two continuation labels by whether promotion is inevitable, unreachable, or action-sensitive at the max horizon.</p>
    <section class="cards">
      <div class="card"><div class="label">Labels</div><div class="value">{cell(summary.get('labels', 0))}</div></div>
      <div class="card"><div class="label">Action Sensitive</div><div class="value">{cell(summary.get('action_sensitive_labels', 0))}</div></div>
      <div class="card"><div class="label">Positive Gains</div><div class="value">{cell(summary.get('positive_promotion_gains', 0))}</div></div>
      <div class="card"><div class="label">Max Gain</div><div class="value">{float(summary.get('max_positive_promotion_gain', 0.0)):.3f}</div></div>
    </section>
    <section class="panel">
      <table><thead><tr><th>Bucket</th><th>Seed</th><th>Move</th><th>To Promo</th><th>Actions</th><th>Promotion Rate</th><th>Span</th><th>Gain</th><th>Regret</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
    </section>
    <section class="panel" style="margin-top:14px;"><pre>{escape(json.dumps(summary, indent=2, sort_keys=True))}</pre></section>
  </main>
</body>
</html>
"""
    path.write_text(html)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    labels = _load_labels(args.labels_json)
    rows = [classify_label(item, rate_gap=args.rate_gap) for item in labels]
    rows.sort(key=lambda row: (str(row.get("bucket")), -float(row.get("promotion_rate_span") or 0.0)))
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_json": str(args.labels_json),
        "rate_gap": float(args.rate_gap),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload["json"] = str(args.out_dir / "promotion_label_report.json")
    payload["html"] = str(args.out_dir / "promotion_label_report.html")
    write_json(args.out_dir / "promotion_label_report.json", payload)
    write_html(args.out_dir / "promotion_label_report.html", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--rate-gap", type=float, default=0.125)
    parser.add_argument("--out-dir", type=Path, default=Path("threes_rl/runs/forensics/promotion_labels/latest"))
    args = parser.parse_args()
    payload = run_from_args(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"json={payload['json']}")
    print(f"html={payload['html']}")


if __name__ == "__main__":
    main()

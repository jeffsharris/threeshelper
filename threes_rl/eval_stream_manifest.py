"""Freeze deterministic, disjoint RNG stream blocks for paired evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from threes_rl.run_artifacts import write_json


EVALUATOR_VERSION = "split_exogenous_v1"
DEFAULT_BLOCK_SIZES = {"D0": 64, "D1": 192, "C": 512}


def stream_id(namespace: str, block: str, index: int, kind: str) -> int:
    raw = f"{namespace}:{block}:{index}:{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def build_stream_manifest(
    *,
    namespace: str,
    block_sizes: dict[str, int] | None = None,
    starter_tile: int | None = 1536,
    logical_seed_start: int = 4_000_000,
) -> dict[str, Any]:
    sizes = dict(DEFAULT_BLOCK_SIZES if block_sizes is None else block_sizes)
    blocks: dict[str, list[dict[str, object]]] = {}
    logical_seed = int(logical_seed_start)
    all_ids: set[int] = set()
    for block, size in sizes.items():
        rows: list[dict[str, object]] = []
        for index in range(int(size)):
            ids = {
                kind: stream_id(namespace, block, index, kind)
                for kind in ("deck", "slot", "policy")
            }
            if any(value in all_ids for value in ids.values()):
                raise RuntimeError("RNG stream ID collision")
            all_ids.update(ids.values())
            rows.append(
                {
                    "index": index,
                    "logical_seed": logical_seed,
                    "starter_tile": starter_tile,
                    "deck_stream_id": ids["deck"],
                    "slot_stream_id": ids["slot"],
                    "policy_stream_id": ids["policy"],
                }
            )
            logical_seed += 1
        blocks[block] = rows
    return {
        "manifest_version": "r1_paired_normal_start_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "namespace": namespace,
        "normal_start_only": True,
        "slot_coupling": "share uniform stream; map independently over each trajectory's legal insertion slots",
        "policy_coupling": "same stream ID, independent generator instance per arm",
        "blocks": blocks,
        "block_sizes": sizes,
        "confirmation_status": "frozen_untouched",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", default="threes-r1-20260709")
    args = parser.parse_args()
    payload = build_stream_manifest(namespace=args.namespace)
    write_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "block_sizes": payload["block_sizes"]}, indent=2))


if __name__ == "__main__":
    main()

"""Initialize an exact frozen-incumbent plus zero staged-residual checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from threes_rl.ntuple import NtupleValue, ResidualStagedNtupleValue
from threes_rl.run_artifacts import write_json
from threes_rl.train_td import TDConfig, create_value_model, save_checkpoint


def _policy_from_file(path: Path) -> str:
    return next(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def initialize(*, run_name: str, policy_file: Path, restart_manifest: Path, seed: int) -> Path:
    policy = _policy_from_file(policy_file)
    run_dir = Path("threes_rl/runs") / run_name
    checkpoint = run_dir / "latest"
    config = TDConfig(
        run_name=run_name,
        games=0,
        pattern_set="default",
        stage_mode="phase4",
        alpha=0.001,
        seed=seed,
        starter_tile=1536,
        max_moves=5000,
        progress_every=100,
        checkpoint_every=0,
        keep_top_games=3,
        actor_policy=policy,
        target_mode="nstep",
        n_step=8,
        use_tc=True,
        start_state_replays=[str(restart_manifest)],
        start_state_prob=0.5,
        start_state_min_tile=0,
        start_state_sample_mode="ancestry_balanced",
        stage_weight_promotion=True,
        promotion_copy_tc=True,
        exact_start_mix=True,
        frozen_incumbent_policy=policy,
        actor_generation_jobs=8,
    )
    if checkpoint.exists():
        meta = json.loads((checkpoint / "meta.json").read_text())
        if int(meta.get("games_completed", -1)) != 0:
            raise FileExistsError(f"R1b initializer refuses nonzero existing checkpoint: {checkpoint}")
        model = NtupleValue.load(checkpoint)
    else:
        model = create_value_model(config)
        if not isinstance(model, ResidualStagedNtupleValue):
            raise TypeError("R1b initializer did not create a residual composite")
        model.enable_temporal_coherence()
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True))
        save_checkpoint(model, run_dir, "latest", config, 0)
    if not isinstance(model, ResidualStagedNtupleValue):
        raise TypeError("R1b initializer checkpoint is not a residual composite")
    frozen_arrays = model.frozen_arrays
    residual_arrays = [
        array
        for stage in model.residual.stages
        for array in [*stage.tables, *(stage.tc_sum_tables or []), *(stage.tc_abs_tables or [])]
    ]
    payload = {
        "decision": "IDENTITY_READY",
        "checkpoint": str(checkpoint),
        "frozen_policy": policy,
        "frozen_components": len(model.frozen_models),
        "frozen_source_fingerprint": model.frozen_source_fingerprint(),
        "frozen_arrays_read_only": all(not array.flags.writeable for array in frozen_arrays),
        "residual_arrays": len(residual_arrays),
        "residual_nonzero_entries": int(sum(np.count_nonzero(array) for array in residual_arrays)),
        "promotion_counts": model.residual.promotion_counts,
        "stage_metrics": model.stage_metrics(),
    }
    write_json(run_dir / "identity_initialization.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--restart-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    initialize(
        run_name=args.run_name,
        policy_file=args.policy_file,
        restart_manifest=args.restart_manifest,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

"""Supervised policy warm-start from a baseline expert."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from threes_rl.baselines import GreedyPolicy
from threes_rl.expectimax import ExpectimaxPolicy
from threes_rl.obs import encode_observation, observation_size
from threes_rl.sim import ThreesSim
from threes_rl.train_ppo import ActorCritic, TrainConfig, save_checkpoint, select_device


def make_expert(name: str):
    if name == "greedy":
        return GreedyPolicy()
    if name == "expectimax2":
        return ExpectimaxPolicy(depth=2)
    raise ValueError(f"Unsupported expert: {name}")


def generate_dataset(
    expert_name: str,
    samples: int,
    seed: int,
    obs_encoder: str,
    starter_tile: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert = make_expert(expert_name)
    rng = np.random.default_rng(seed)
    obs_dim = observation_size(obs_encoder)
    obs = np.zeros((samples, obs_dim), dtype=np.float32)
    actions = np.zeros(samples, dtype=np.int64)
    masks = np.zeros((samples, 4), dtype=bool)
    filled = 0
    episode = 0
    start = time.perf_counter()
    while filled < samples:
        sim = ThreesSim(np.random.default_rng(seed + 10_000 + episode), starter_tile=starter_tile)
        policy_rng = np.random.default_rng(seed + 20_000 + episode)
        state = sim.reset()
        while not state.game_over and filled < samples:
            legal_mask = sim.legal_mask(state)
            action = int(expert(state, sim, policy_rng))
            obs[filled] = encode_observation(state, sim, obs_encoder)
            actions[filled] = action
            masks[filled] = legal_mask
            state, info = sim.step(state, action)
            if not info.moved:
                break
            filled += 1
            if filled % 5000 == 0:
                elapsed = max(1e-9, time.perf_counter() - start)
                print(f"generated {filled}/{samples} samples ({filled / elapsed:.1f}/s)", flush=True)
        episode += 1
    return obs, actions, masks


def train_imitation(
    *,
    run_name: str,
    expert: str,
    samples: int,
    epochs: int,
    batch_size: int,
    seed: int,
    obs_encoder: str,
    starter_tile: Optional[int],
    device_name: str,
) -> Path:
    run_dir = Path("threes_rl/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    obs_dim = observation_size(obs_encoder)
    obs, actions, masks = generate_dataset(expert, samples, seed, obs_encoder, starter_tile)
    np.savez_compressed(run_dir / "dataset_summary.npz", actions=actions[: min(samples, 1000)])

    model = ActorCritic(obs_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    indices = np.arange(samples)
    metrics_path = run_dir / "imitation_metrics.jsonl"
    for epoch in range(1, epochs + 1):
        np.random.default_rng(seed + epoch).shuffle(indices)
        losses = []
        accuracies = []
        for start_idx in range(0, samples, batch_size):
            batch_idx_np = indices[start_idx : start_idx + batch_size]
            batch_obs = torch.as_tensor(obs[batch_idx_np], dtype=torch.float32, device=device)
            batch_actions = torch.as_tensor(actions[batch_idx_np], dtype=torch.int64, device=device)
            batch_masks = torch.as_tensor(masks[batch_idx_np], dtype=torch.bool, device=device)
            logits, _values = model(batch_obs)
            logits = logits.masked_fill(~batch_masks, -1e9)
            loss = loss_fn(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                accuracies.append(float((pred == batch_actions).float().mean().item()))
                losses.append(float(loss.item()))
        metrics = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accuracies)),
            "samples": samples,
            "expert": expert,
        }
        print(json.dumps(metrics, sort_keys=True))
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(metrics, sort_keys=True) + "\n")

    config = TrainConfig(
        run_name=run_name,
        total_steps=0,
        num_envs=0,
        rollout_steps=0,
        seed=seed,
        obs_encoder=obs_encoder,
        reward_mode="final_score",
        starter_tile=starter_tile,
        obs_dim=obs_dim,
    )
    with (run_dir / "config.json").open("w") as fh:
        json.dump({**asdict(config), "imitation_expert": expert, "imitation_samples": samples}, fh, indent=2, sort_keys=True)
    save_checkpoint(run_dir / "latest.pt", model, optimizer, config, env_steps=0, update=epochs)
    return run_dir / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"imitation_{int(time.time())}")
    parser.add_argument("--expert", choices=["greedy", "expectimax2"], default="greedy")
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obs-encoder", default="full")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    starter_tile = None if args.starter.lower() == "none" else int(args.starter)
    checkpoint = train_imitation(
        run_name=args.run_name,
        expert=args.expert,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        obs_encoder=args.obs_encoder,
        starter_tile=starter_tile,
        device_name=args.device,
    )
    print(f"latest_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

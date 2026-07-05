"""Supervised policy warm-start from a baseline expert."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
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
    *,
    progress_every: int = 5000,
    progress_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert = make_expert(expert_name)
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
            if progress_every > 0 and filled % progress_every == 0:
                elapsed = max(1e-9, time.perf_counter() - start)
                print(f"{progress_prefix}generated {filled}/{samples} samples ({filled / elapsed:.1f}/s)", flush=True)
        episode += 1
    return obs, actions, masks


def _split_sample_counts(samples: int, workers: int, chunk_size: int) -> list[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if workers <= 1:
        return [samples]
    chunk_size = max(1, chunk_size)
    counts: list[int] = []
    remaining = samples
    while remaining > 0:
        counts.append(min(chunk_size, remaining))
        remaining -= counts[-1]
    return counts


def _generate_dataset_chunk(args: tuple[str, int, int, str, Optional[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_name, samples, seed, obs_encoder, starter_tile = args
    return generate_dataset(
        expert_name,
        samples,
        seed,
        obs_encoder,
        starter_tile,
        progress_every=0,
    )


def generate_dataset_parallel(
    expert_name: str,
    samples: int,
    seed: int,
    obs_encoder: str,
    starter_tile: Optional[int],
    *,
    workers: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if workers <= 1:
        return generate_dataset(expert_name, samples, seed, obs_encoder, starter_tile)

    counts = _split_sample_counts(samples, workers, chunk_size)
    max_workers = min(workers, len(counts))
    started = time.perf_counter()
    print(f"generating {samples} samples across {max_workers} workers in {len(counts)} chunks", flush=True)
    tasks = [
        (expert_name, count, seed + 1_000_003 * (idx + 1), obs_encoder, starter_tile)
        for idx, count in enumerate(counts)
    ]
    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * len(tasks)
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        future_to_idx = {executor.submit(_generate_dataset_chunk, task): idx for idx, task in enumerate(tasks)}
        completed_samples = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk = future.result()
            chunks[idx] = chunk
            completed_samples += len(chunk[1])
            elapsed = max(1e-9, time.perf_counter() - started)
            print(f"generated {completed_samples}/{samples} samples ({completed_samples / elapsed:.1f}/s)", flush=True)

    ready_chunks = [chunk for chunk in chunks if chunk is not None]
    return (
        np.concatenate([chunk[0] for chunk in ready_chunks], axis=0),
        np.concatenate([chunk[1] for chunk in ready_chunks], axis=0),
        np.concatenate([chunk[2] for chunk in ready_chunks], axis=0),
    )


def save_dataset(
    path: Path,
    obs: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    *,
    expert: str,
    obs_encoder: str,
    starter_tile: Optional[int],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs=obs,
        actions=actions,
        masks=masks,
        expert=np.asarray(expert),
        obs_encoder=np.asarray(obs_encoder),
        starter_tile=np.asarray(-1 if starter_tile is None else starter_tile, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
    )


def load_dataset(path: Path, samples: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        obs = np.asarray(payload["obs"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.int64)
        masks = np.asarray(payload["masks"], dtype=bool)
    if obs.ndim != 2:
        raise ValueError(f"Dataset obs must be rank 2, got shape {obs.shape}")
    if actions.ndim != 1:
        raise ValueError(f"Dataset actions must be rank 1, got shape {actions.shape}")
    if masks.shape != (len(actions), 4):
        raise ValueError(f"Dataset masks must have shape ({len(actions)}, 4), got {masks.shape}")
    if len(obs) != len(actions):
        raise ValueError(f"Dataset obs/actions row mismatch: {len(obs)} != {len(actions)}")
    if samples is not None:
        if samples > len(actions):
            raise ValueError(f"Requested {samples} samples but dataset only has {len(actions)}")
        obs = obs[:samples]
        actions = actions[:samples]
        masks = masks[:samples]
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
    workers: int,
    chunk_size: int,
    dataset_path: Optional[Path],
    save_full_dataset: bool,
    checkpoint_every: int,
) -> Path:
    run_dir = Path("threes_rl/runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    obs_dim = observation_size(obs_encoder)
    dataset_source = "generated"
    if dataset_path is not None and dataset_path.exists():
        obs, actions, masks = load_dataset(dataset_path, samples=samples)
        if obs.shape[1] != obs_dim:
            raise ValueError(f"Dataset obs width {obs.shape[1]} does not match encoder {obs_encoder!r} width {obs_dim}")
        samples = len(actions)
        dataset_source = str(dataset_path)
        print(f"loaded {samples} samples from {dataset_path}", flush=True)
    else:
        obs, actions, masks = generate_dataset_parallel(
            expert,
            samples,
            seed,
            obs_encoder,
            starter_tile,
            workers=workers,
            chunk_size=chunk_size,
        )
        if dataset_path is not None:
            save_dataset(dataset_path, obs, actions, masks, expert=expert, obs_encoder=obs_encoder, starter_tile=starter_tile, seed=seed)
            dataset_source = str(dataset_path)
            print(f"saved full dataset to {dataset_path}", flush=True)
        elif save_full_dataset:
            saved_path = run_dir / "dataset.npz"
            save_dataset(saved_path, obs, actions, masks, expert=expert, obs_encoder=obs_encoder, starter_tile=starter_tile, seed=seed)
            dataset_source = str(saved_path)
            print(f"saved full dataset to {saved_path}", flush=True)
    np.savez_compressed(run_dir / "dataset_summary.npz", actions=actions[: min(samples, 1000)])

    config = TrainConfig(
        run_name=run_name,
        total_steps=0,
        num_envs=0,
        rollout_steps=0,
        seed=seed,
        obs_encoder=obs_encoder,
        reward_mode="final_score",
        starter_tile=starter_tile,
        device=device_name,
        obs_dim=obs_dim,
    )
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
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch}.pt", model, optimizer, config, env_steps=0, update=epoch)
            save_checkpoint(run_dir / "latest.pt", model, optimizer, config, env_steps=0, update=epoch)

    with (run_dir / "config.json").open("w") as fh:
        json.dump({**asdict(config), "imitation_expert": expert, "imitation_samples": samples}, fh, indent=2, sort_keys=True)
    metadata = {
        "dataset_source": dataset_source,
        "expert": expert,
        "samples": samples,
        "obs_encoder": obs_encoder,
        "starter_tile": starter_tile,
        "seed": seed,
        "workers": workers,
        "chunk_size": chunk_size,
        "checkpoint_every": checkpoint_every,
    }
    with (run_dir / "dataset_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)
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
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--dataset-path", type=Path, help="Load an existing dataset, or save a generated dataset if the path is absent.")
    parser.add_argument("--save-full-dataset", action="store_true", help="Save generated obs/actions/masks to the run directory.")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save checkpoint_epoch_<N>.pt every N imitation epochs.")
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
        workers=args.workers,
        chunk_size=args.chunk_size,
        dataset_path=args.dataset_path,
        save_full_dataset=args.save_full_dataset,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"latest_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

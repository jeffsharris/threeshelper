"""PPO training entrypoint with invalid-action masking."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from torch import nn
from torch.distributions import Categorical

from threes_rl.env import ThreesEnv
from threes_rl.obs import observation_size


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, 4)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(obs)
        return self.policy(features), self.value(features).squeeze(-1)


@dataclass
class TrainConfig:
    run_name: str
    total_steps: int = 1_000_000
    num_envs: int = 64
    rollout_steps: int = 128
    gamma: float = 1.0
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 4096
    update_epochs: int = 4
    seed: int = 1
    obs_encoder: str = "full"
    reward_mode: str = "final_score"
    reward_scale: float = 1e-5
    starter_tile: Optional[int] = 1536
    checkpoint_interval: int = 1_000_000
    device: str = "auto"
    obs_dim: int = 0


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_env(seed: int, config: TrainConfig):
    def _factory():
        return ThreesEnv(
            seed=seed,
            starter_tile=config.starter_tile,
            obs_encoder=config.obs_encoder,
            reward_mode=config.reward_mode,
        )

    return _factory


def legal_mask_from_infos(infos: dict) -> np.ndarray:
    mask = np.asarray(infos["legal_mask"], dtype=bool).copy()
    no_legal = ~mask.any(axis=1)
    if np.any(no_legal):
        mask[no_legal, 0] = True
    return mask


def sample_actions(model: ActorCritic, obs: np.ndarray, legal_mask: np.ndarray, device: torch.device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
    logits, values = model(obs_t)
    logits = logits.masked_fill(~mask_t, -1e9)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    return actions, dist.log_prob(actions), dist.entropy(), values


def save_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer, config: TrainConfig, env_steps: int, update: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "env_steps": env_steps,
            "update": update,
        },
        path,
    )


def load_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, int]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("env_steps", 0)), int(payload.get("update", 0))


def train(config: TrainConfig, resume: Optional[Path] = None) -> Path:
    config.obs_dim = observation_size(config.obs_encoder)
    run_dir = Path("threes_rl/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as fh:
        json.dump(asdict(config), fh, indent=2, sort_keys=True)

    device = select_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = SyncVectorEnv([make_env(config.seed + idx, config) for idx in range(config.num_envs)])
    obs, infos = envs.reset(seed=config.seed)

    model = ActorCritic(config.obs_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    start_steps = 0
    start_update = 0
    if resume is not None:
        start_steps, start_update = load_checkpoint(resume, model, optimizer, device)

    rollout_size = config.num_envs * config.rollout_steps
    num_updates = max(1, (config.total_steps - start_steps) // rollout_size)
    metrics_path = run_dir / "metrics.jsonl"

    episode_scores: list[float] = []
    episode_moves: list[float] = []
    episode_max_tiles: list[int] = []
    env_steps = start_steps
    last_checkpoint = start_steps
    start_time = time.perf_counter()

    for update_idx in range(start_update + 1, start_update + num_updates + 1):
        frac = 1.0 - (update_idx - start_update - 1.0) / max(1, num_updates)
        optimizer.param_groups[0]["lr"] = frac * config.learning_rate

        obs_buf = np.zeros((config.rollout_steps, config.num_envs, config.obs_dim), dtype=np.float32)
        action_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.int64)
        logprob_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        reward_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        done_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        value_buf = np.zeros((config.rollout_steps, config.num_envs), dtype=np.float32)
        mask_buf = np.zeros((config.rollout_steps, config.num_envs, 4), dtype=bool)

        for step in range(config.rollout_steps):
            legal_mask = legal_mask_from_infos(infos)
            with torch.no_grad():
                actions, logprobs, _entropy, values = sample_actions(model, obs, legal_mask, device)
            obs_buf[step] = obs
            action_buf[step] = actions.cpu().numpy()
            logprob_buf[step] = logprobs.cpu().numpy()
            value_buf[step] = values.cpu().numpy()
            mask_buf[step] = legal_mask

            obs, rewards, terminated, truncated, infos = envs.step(action_buf[step])
            dones = np.logical_or(terminated, truncated)
            reward_buf[step] = (rewards * config.reward_scale).astype(np.float32)
            done_buf[step] = dones.astype(np.float32)
            env_steps += config.num_envs

            final_mask = np.asarray(infos.get("_final_score", np.zeros(config.num_envs, dtype=bool)), dtype=bool)
            if "final_score" in infos:
                for score in np.asarray(infos["final_score"])[final_mask]:
                    episode_scores.append(float(score))
            if "move_count" in infos:
                move_mask = np.logical_and(final_mask, np.asarray(infos.get("_move_count", final_mask), dtype=bool))
                for moves in np.asarray(infos["move_count"])[move_mask]:
                    episode_moves.append(float(moves))
            if "max_tile" in infos:
                max_mask = np.logical_and(final_mask, np.asarray(infos.get("_max_tile", final_mask), dtype=bool))
                for max_tile in np.asarray(infos["max_tile"])[max_mask]:
                    episode_max_tiles.append(int(max_tile))

        next_mask = legal_mask_from_infos(infos)
        with torch.no_grad():
            _next_actions, _next_logprobs, _next_entropy, next_values_t = sample_actions(model, obs, next_mask, device)
        next_values = next_values_t.cpu().numpy()

        advantages = np.zeros_like(reward_buf, dtype=np.float32)
        lastgaelam = np.zeros(config.num_envs, dtype=np.float32)
        for t in reversed(range(config.rollout_steps)):
            if t == config.rollout_steps - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_value = next_values
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_value = value_buf[t + 1]
            delta = reward_buf[t] + config.gamma * next_value * next_nonterminal - value_buf[t]
            lastgaelam = delta + config.gamma * config.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + value_buf

        b_obs = torch.as_tensor(obs_buf.reshape((-1, config.obs_dim)), dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(action_buf.reshape(-1), dtype=torch.int64, device=device)
        b_logprobs = torch.as_tensor(logprob_buf.reshape(-1), dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        b_values = torch.as_tensor(value_buf.reshape(-1), dtype=torch.float32, device=device)
        b_masks = torch.as_tensor(mask_buf.reshape((-1, 4)), dtype=torch.bool, device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        batch_size = rollout_size
        minibatch_size = min(config.minibatch_size, batch_size)
        batch_indices = np.arange(batch_size)
        last_loss = {}
        for _epoch in range(config.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = torch.as_tensor(batch_indices[start : start + minibatch_size], dtype=torch.int64, device=device)
                logits, new_values = model(b_obs[mb_idx])
                logits = logits.masked_fill(~b_masks[mb_idx], -1e9)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(b_actions[mb_idx])
                entropy = dist.entropy().mean()
                logratio = new_logprobs - b_logprobs[mb_idx]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_idx]
                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_loss = 0.5 * ((new_values - b_returns[mb_idx]) ** 2).mean()
                loss = policy_loss - config.entropy_coef * entropy + config.value_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                last_loss = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl),
                }

        recent_scores = episode_scores[-100:]
        recent_moves = episode_moves[-100:]
        recent_max = episode_max_tiles[-100:]
        metrics = {
            "update": update_idx,
            "env_steps": env_steps,
            "steps_per_s": env_steps / max(1e-9, time.perf_counter() - start_time),
            "mean_episode_score_100": float(np.mean(recent_scores)) if recent_scores else None,
            "mean_episode_moves_100": float(np.mean(recent_moves)) if recent_moves else None,
            "max_tile_ge_192": float(np.mean([tile >= 192 for tile in recent_max])) if recent_max else None,
            "max_tile_ge_384": float(np.mean([tile >= 384 for tile in recent_max])) if recent_max else None,
            "max_tile_ge_768": float(np.mean([tile >= 768 for tile in recent_max])) if recent_max else None,
            "max_tile_ge_1536": float(np.mean([tile >= 1536 for tile in recent_max])) if recent_max else None,
            **last_loss,
        }
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(json.dumps(metrics, sort_keys=True))

        if env_steps - last_checkpoint >= config.checkpoint_interval or env_steps >= config.total_steps:
            save_checkpoint(run_dir / f"checkpoint_{env_steps}.pt", model, optimizer, config, env_steps, update_idx)
            save_checkpoint(run_dir / "latest.pt", model, optimizer, config, env_steps, update_idx)
            last_checkpoint = env_steps

    save_checkpoint(run_dir / f"checkpoint_{env_steps}.pt", model, optimizer, config, env_steps, update_idx)
    save_checkpoint(run_dir / "latest.pt", model, optimizer, config, env_steps, update_idx)
    envs.close()
    return run_dir / "latest.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=f"ppo_{int(time.time())}")
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--obs-encoder", default="full")
    parser.add_argument("--reward-mode", default="final_score")
    parser.add_argument("--reward-scale", type=float, default=1e-5)
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    starter_tile = None if args.starter.lower() == "none" else int(args.starter)
    config = TrainConfig(
        run_name=args.run_name,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        seed=args.seed,
        obs_encoder=args.obs_encoder,
        reward_mode=args.reward_mode,
        reward_scale=args.reward_scale,
        starter_tile=starter_tile,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
    )
    checkpoint = train(config, resume=args.resume)
    print(f"latest_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

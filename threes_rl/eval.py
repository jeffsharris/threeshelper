"""Evaluate Threes policies on deterministic seed suites."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

import numpy as np

from threes_rl.baselines import GreedyPolicy, RandomPolicy
from threes_rl.expectimax import ExpectimaxPolicy
from threes_rl.obs import encode_observation
from threes_rl.sim import SimState, ThreesSim, score_board


@dataclass
class GameResult:
    seed: int
    score: int
    moves: int
    max_tile: int
    terminal_tile: bool


class PpoPolicy:
    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        import torch

        from threes_rl.train_ppo import ActorCritic

        payload = torch.load(checkpoint, map_location=device)
        config = payload["config"]
        self.device = torch.device(device)
        self.obs_encoder = config.get("obs_encoder", "full")
        self.model = ActorCritic(int(config["obs_dim"])).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.name = f"ppo:{checkpoint}"

    def __call__(self, state: SimState, sim: ThreesSim, rng: np.random.Generator) -> int:
        import torch

        obs = encode_observation(state, sim, self.obs_encoder)
        mask = sim.legal_mask(state)
        with torch.no_grad():
            logits, _value = self.model(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            logits = logits.squeeze(0)
            logits[~torch.as_tensor(mask, dtype=torch.bool, device=self.device)] = -1e9
            return int(torch.argmax(logits).item())


def parse_seed_range(text: str) -> list[int]:
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end)))
    return [int(part) for part in text.split(",") if part]


def make_policy(spec: str):
    if spec == "random":
        return RandomPolicy()
    if spec == "greedy":
        return GreedyPolicy()
    if spec == "expectimax2":
        return ExpectimaxPolicy(depth=2)
    if spec == "expectimax3":
        return ExpectimaxPolicy(depth=3)
    if spec.startswith("ppo:"):
        return PpoPolicy(Path(spec.split(":", 1)[1]))
    raise ValueError(f"Unsupported policy: {spec}")


def run_game(policy, seed: int, starter_tile: int | None, max_moves: int) -> GameResult:
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter_tile)
    policy_rng = np.random.default_rng(seed + 1_000_003)
    state = sim.reset()
    while not state.game_over and state.move_count < max_moves:
        action = int(policy(state, sim, policy_rng))
        state, info = sim.step(state, action)
        if not info.moved:
            legal = sim.legal_actions(state)
            if not legal:
                break
            state, _info = sim.step(state, legal[0])
    return GameResult(
        seed=seed,
        score=score_board(state.board),
        moves=state.move_count,
        max_tile=state.max_tile,
        terminal_tile=bool(np.any(state.board == 12288)),
    )


def summarize(results: list[GameResult]) -> dict[str, object]:
    scores = sorted(result.score for result in results)
    if not scores:
        raise ValueError("No results to summarize")
    p90_idx = min(len(scores) - 1, int(0.9 * (len(scores) - 1)))
    thresholds = [192, 384, 768, 1536, 3072, 6144, 12288]
    max_tile_dist = {f">={threshold}": sum(1 for result in results if result.max_tile >= threshold) / len(results) for threshold in thresholds}
    return {
        "games": len(results),
        "mean_score": mean(scores),
        "median_score": median(scores),
        "p90_score": scores[p90_idx],
        "mean_moves": mean(result.moves for result in results),
        "max_tile_dist": max_tile_dist,
    }


def append_results(policy_name: str, command: str, summary: dict[str, object]) -> None:
    path = Path("threes_rl/RESULTS.md")
    with path.open("a") as fh:
        fh.write(f"\n## Eval: {policy_name}\n\n")
        fh.write(f"Command: `{command}`\n\n")
        fh.write("```json\n")
        fh.write(json.dumps(summary, indent=2, sort_keys=True))
        fh.write("\n```\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="random, greedy, expectimax2, expectimax3, or ppo:<checkpoint>")
    parser.add_argument("--seeds", default="1000:1200")
    parser.add_argument("--starter", default="1536")
    parser.add_argument("--max-moves", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--no-append", action="store_true")
    args = parser.parse_args()

    starter = None if args.starter.lower() == "none" else int(args.starter)
    seeds = parse_seed_range(args.seeds)
    policy = make_policy(args.policy)
    results = []
    for idx, seed in enumerate(seeds, start=1):
        results.append(run_game(policy, seed, starter, args.max_moves))
        if args.progress_every and idx % args.progress_every == 0:
            partial = summarize(results)
            print(
                f"progress {idx}/{len(seeds)} "
                f"mean_score={partial['mean_score']:.2f} "
                f"mean_moves={partial['mean_moves']:.2f}",
                flush=True,
            )
    summary = summarize(results)

    out_dir = Path("threes_rl/runs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.policy.replace(':', '_').replace('/', '_')}_{seeds[0]}_{seeds[-1]}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed", "score", "moves", "max_tile", "terminal_tile"])
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"per_seed_csv={csv_path}")
    if not args.no_append:
        append_results(args.policy, "python -m threes_rl.eval " + " ".join(_quote_args()), summary)


def _quote_args() -> list[str]:
    import sys

    return sys.argv[1:]


if __name__ == "__main__":
    main()

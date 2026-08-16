"""Play the Threes simulator from the terminal for sanity checks."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from threes_rl.sim import DIRECTION_NAMES, ThreesSim, score_board

KEY_TO_ACTION = {
    "w": 0,
    "k": 0,
    "up": 0,
    "s": 1,
    "j": 1,
    "down": 1,
    "a": 2,
    "h": 2,
    "left": 2,
    "d": 3,
    "l": 3,
    "right": 3,
}


def format_preview(state) -> str:
    preview = state.preview
    if preview.kind == "bonus":
        return f"large {list(preview.candidates)}"
    return f"{preview.label} ({preview.value})"


def render(sim: ThreesSim, state) -> str:
    rows = []
    for row in state.board.tolist():
        rows.append(" ".join(f"{value:>5}" if value else "    ." for value in row))
    legal = ", ".join(DIRECTION_NAMES[action] for action in sim.legal_actions(state)) or "-"
    return "\n".join(
        [
            "",
            *rows,
            "",
            f"score={score_board(state.board)} moves={state.move_count} max={state.max_tile}",
            f"next={format_preview(state)} legal={legal}",
            "move: w/a/s/d, h/j/k/l, arrow name, r reset, q quit",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--starter", default="1536")
    args = parser.parse_args()

    starter = None if args.starter.lower() == "none" else int(args.starter)
    seed = int(args.seed)
    sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter)
    state = sim.reset()
    while True:
        print(render(sim, state), flush=True)
        if state.game_over or not sim.legal_actions(state):
            print("Game over. Type r to reset or q to quit.", flush=True)
        raw = input("> ").strip().lower()
        if raw in ("q", "quit", "exit"):
            return
        if raw in ("r", "reset"):
            seed += 1
            sim = ThreesSim(np.random.default_rng(seed), starter_tile=starter)
            state = sim.reset()
            continue
        action = KEY_TO_ACTION.get(raw)
        if action is None:
            print(f"Unknown input: {raw}", file=sys.stderr)
            continue
        state, info = sim.step(state, action)
        if not info.moved:
            print("Illegal/no-op move.", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Pygame application launcher."""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import load_config
from utils.reproducibility import seed_everything
from visualization.controller import App


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the AlphaZero Connect Four desktop app.")
    parser.add_argument("--mode", choices=["menu", "human", "ai"], default="menu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-b", default=None)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_b = Path(args.checkpoint_b) if args.checkpoint_b else None
    if checkpoint and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if checkpoint_b and not checkpoint_b.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_b}")
    app = App(
        config,
        mode=args.mode,
        checkpoint=checkpoint,
        checkpoint_b=checkpoint_b,
        simulations=args.simulations,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

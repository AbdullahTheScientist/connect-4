"""Evaluate a checkpoint against another network, a random agent, or the tactical baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from evaluation.arena import Arena, MCTSAgent
from evaluation.baseline import RandomAgent, TacticalAgent
from evaluation.metrics import save_evaluation_result
from model.checkpoint import load_model
from utils.config import PROJECT_ROOT, load_config, resolve_device, resolve_path
from utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AlphaZero Connect Four checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument(
        "--opponent",
        default="random",
        help="random, tactical, initial, best, or a checkpoint path (use --opponent-checkpoint)",
    )
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None, help="JSON result file (appended)")
    parser.add_argument("--charts", action="store_true", help="Write a Matplotlib result chart")
    return parser.parse_args()


def resolve_sibling(checkpoint: Path, name: str) -> Path:
    sibling = checkpoint.parent / name
    if sibling.exists():
        return sibling
    direct = checkpoint.with_name(name)
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Could not find {name} next to {checkpoint}")


def plot_result(result, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#111927")
    ax.set_facecolor("#172233")
    labels = ["wins", "losses", "draws"]
    values = [result.wins, result.losses, result.draws]
    bars = ax.bar(labels, values, color=["#4bcd8b", "#f05658", "#f6be5c"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(value), ha="center", color="#e9f0f8")
    ax.set_title(f"{result.agent_a} vs {result.agent_b}", color="#e9f0f8", loc="left")
    ax.tick_params(colors="#90a1b8")
    for spine in ax.spines.values():
        spine.set_color("#2f3f57")
    ax.grid(axis="y", color="#2a394f", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    simulations = int(args.simulations or config.get("evaluation", {}).get("simulations", 50))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute() and not checkpoint.exists():
        alternative = PROJECT_ROOT / checkpoint
        if alternative.exists():
            checkpoint = alternative
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model, payload = load_model(checkpoint, device=device)
    model_name = f"{checkpoint.parent.name}/{checkpoint.stem}" if checkpoint.parent.name != "checkpoints" else checkpoint.stem
    agent = MCTSAgent(
        model,
        config,
        device=device,
        simulations=simulations,
        name=model_name,
        temperature=0.0,
        seed=args.seed,
    )
    opponent = args.opponent
    if args.opponent_checkpoint:
        opponent_path = Path(args.opponent_checkpoint)
        if not opponent_path.is_absolute() and not opponent_path.exists():
            opponent_path = PROJECT_ROOT / opponent_path
    elif opponent.lower() == "initial":
        opponent_path = resolve_sibling(checkpoint, "initial.pt")
    elif opponent.lower() == "best":
        opponent_path = resolve_sibling(checkpoint, "best.pt")
    elif opponent.lower() not in ("random", "tactical"):
        opponent_path = Path(opponent)
        if not opponent_path.is_absolute() and not opponent_path.exists():
            opponent_path = PROJECT_ROOT / opponent_path
    else:
        opponent_path = None

    arena = Arena(config)
    progress = tqdm(total=args.games, desc="evaluation", unit="game")

    def callback(done, total, partial):
        progress.update(1)
        progress.set_postfix(w=partial.wins, l=partial.losses, d=partial.draws)

    start = time.perf_counter()
    if opponent.lower() == "random":
        result = arena.evaluate(agent, RandomAgent(seed=args.seed + 5), games=args.games, seed=args.seed, simulations=simulations, progress_callback=callback)
    elif opponent.lower() == "tactical":
        result = arena.evaluate(agent, TacticalAgent(seed=args.seed + 9), games=args.games, seed=args.seed, simulations=simulations, progress_callback=callback)
    else:
        opponent_model, _ = load_model(opponent_path, device=device)
        opponent_agent = MCTSAgent(
            opponent_model,
            config,
            device=device,
            simulations=simulations,
            name=f"{opponent_path.parent.name}/{opponent_path.stem}",
            temperature=0.0,
            seed=args.seed + 1,
        )
        result = arena.evaluate(agent, opponent_agent, games=args.games, seed=args.seed, simulations=simulations, progress_callback=callback)
    progress.close()
    if result.elapsed_seconds == 0:
        result.elapsed_seconds = time.perf_counter() - start
    result.metadata.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_iteration": payload.get("iteration", 0),
            "opponent": opponent,
            "opponent_checkpoint": str(opponent_path) if opponent_path else opponent,
            "device": str(device),
            "seed": args.seed,
        }
    )
    print(result.summary())
    print(f"Average game length: {result.avg_game_length:.2f} | time: {result.elapsed_seconds:.1f}s")
    output = Path(args.output) if args.output else resolve_path(config, "results") / "cli_evaluations.json"
    save_evaluation_result(result, output, record_type="cli_evaluation")
    print(f"Result appended to {output}")
    if args.charts:
        chart_path = resolve_path(config, "results") / "plots" / f"evaluation_{checkpoint.stem}_{int(time.time())}.png"
        plot_result(result, chart_path)
        print(f"Chart written to {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

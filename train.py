"""Command line entry point for AlphaZero Connect Four training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.trainer import Trainer
from utils.config import load_config, resolve_device
from utils.logging import TrainingStopped
from utils.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AlphaZero-style Connect Four agent through self-play.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--experiment", default=None, help="Experiment name (checkpoint/log subdirectory)")
    parser.add_argument("--iterations", type=int, default=None, help="Total training iteration target")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment seed")
    parser.add_argument("--games", type=int, default=None, help="Override self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=None, help="Override MCTS simulations per move")
    parser.add_argument("--workers", type=int, default=None, help="Self-play worker threads")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per iteration")
    parser.add_argument("--control-file", default=None, help="File-backed pause/stop control path")
    parser.add_argument("--fresh", action="store_true", help="Ignore --resume (explicit fresh start)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    seed_everything(int(config["experiment"]["seed"]), bool(config["experiment"].get("deterministic", True)))

    if args.games is not None:
        config["selfplay"]["games_per_iteration"] = args.games
    if args.simulations is not None:
        config["mcts"]["simulations"] = args.simulations
        config["evaluation"]["simulations"] = args.simulations
    if args.workers is not None:
        config["selfplay"]["workers"] = args.workers
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    experiment = args.experiment
    resume = None if args.fresh else args.resume
    if resume and experiment is None:
        resume_path = Path(resume)
        checkpoints_root = Path(config.get("paths", {}).get("checkpoints", "checkpoints"))
        if not checkpoints_root.is_absolute():
            from utils.config import PROJECT_ROOT

            checkpoints_root = PROJECT_ROOT / checkpoints_root
        try:
            relative = resume_path.resolve().relative_to(checkpoints_root.resolve())
            if len(relative.parts) > 1:
                experiment = relative.parts[0]
        except ValueError:
            pass

    trainer = Trainer(
        config,
        experiment=experiment,
        resume=resume,
        device=str(device),
        control_file=args.control_file,
        iterations=args.iterations,
    )
    try:
        trainer.train()
    except TrainingStopped:
        return 0
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    finally:
        trainer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

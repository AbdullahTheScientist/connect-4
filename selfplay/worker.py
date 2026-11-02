"""Process-safe self-play task. Used for optional multi-process generation."""
from __future__ import annotations

from typing import Any, Dict

import torch

from model.network import AlphaZeroNetwork
from selfplay.generator import GameRecord, SelfPlayGenerator


def generate_game_task(payload: Dict[str, Any]) -> GameRecord:
    """Reconstruct a network in a worker and play one complete self-play game.

    The default generator uses threads (compatible with CUDA). This entry point
    enables true process parallelism on CPU when requested by a custom runner.
    """
    model = AlphaZeroNetwork.from_config(payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    generator = SelfPlayGenerator(
        model,
        payload["config"],
        device=torch.device(payload.get("device", "cpu")),
        simulations=payload.get("simulations"),
    )
    return generator.generate_game(int(payload.get("seed", 0)))


def make_task_payload(
    model: AlphaZeroNetwork,
    config: dict,
    seed: int,
    device: str = "cpu",
    simulations: int | None = None,
) -> Dict[str, Any]:
    return {
        "model_config": model.config(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": config,
        "seed": int(seed),
        "device": device,
        "simulations": simulations,
    }

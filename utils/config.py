"""YAML configuration loading, deep merging and validation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {"name": "default", "seed": 42, "deterministic": True},
    "game": {"rows": 6, "columns": 7, "connect": 4},
    "model": {"channels": 64, "residual_blocks": 4, "input_channels": 2},
    "mcts": {
        "simulations": 50,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "temperature": 1.0,
        "temperature_moves": 12,
    },
    "selfplay": {
        "games_per_iteration": 50,
        "workers": 1,
        "temperature": 1.0,
        "temperature_moves": 12,
        "save_examples": True,
    },
    "training": {
        "iterations": 50,
        "epochs": 5,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "l2_coefficient": 0.0001,
        "gradient_clip": 1.0,
        "replay_capacity": 50000,
        "min_buffer_size": 256,
        "max_batches_per_epoch": 64,
        "lr_step_size": 999,
        "lr_gamma": 0.5,
        "save_buffer": True,
        "checkpoint_every": 1,
    },
    "evaluation": {
        "candidate_games": 20,
        "random_games": 20,
        "simulations": 50,
        "promotion_threshold": 0.55,
        "temperature": 0.0,
        "tactical_games": 20,
    },
    "paths": {
        "checkpoints": "checkpoints",
        "logs": "logs",
        "results": "results",
    },
    "ui": {
        "width": 1280,
        "height": 800,
        "fps": 60,
        "animation_speed": 900,
        "default_simulations": 50,
    },
    "tensorboard": {"enabled": True},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    config = _deep_merge(DEFAULT_CONFIG, data)
    config["_config_path"] = str(path)
    return config


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False)


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    base = Path(config.get("paths", {}).get(key, key))
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base


def training_dir(config: Dict[str, Any], experiment: Optional[str] = None) -> Path:
    name = experiment or config.get("experiment", {}).get("name", "default")
    base = resolve_path(config, "checkpoints")
    if name in (None, "", "default"):
        return base
    return base / name


def log_dir(config: Dict[str, Any], experiment: Optional[str] = None) -> Path:
    name = experiment or config.get("experiment", {}).get("name", "default")
    return resolve_path(config, "logs") / name


def results_dir(config: Dict[str, Any], experiment: Optional[str] = None) -> Path:
    name = experiment or config.get("experiment", {}).get("name", "default")
    return resolve_path(config, "results") / name

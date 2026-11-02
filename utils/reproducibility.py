"""Deterministic seeding helpers used by every experiment entry point."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch without changing global model state."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older torch
            pass


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def rng_from_state(state: Optional[dict]) -> np.random.Generator:
    rng = np.random.default_rng()
    if state:
        rng.bit_generator.state = state
    return rng

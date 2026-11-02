"""Robust checkpoint I/O and checkpoint discovery for the GUI/CLI."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from model.network import AlphaZeroNetwork, count_parameters


FORMAT_VERSION = 1


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        # Validate before replacing a potentially existing best checkpoint.
        check = torch.load(tmp_path, map_location="cpu", weights_only=False)
        required = {"model_state", "model_config", "iteration"}
        if not required.issubset(check):
            raise ValueError("Saved checkpoint failed validation")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def save_checkpoint(
    path: str | Path,
    model: AlphaZeroNetwork,
    optimizer: Optional[torch.optim.Optimizer] = None,
    iteration: int = 0,
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
    model_type: str = "iteration",
    scheduler=None,
) -> Path:
    path = Path(path)
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_type": model_type,
        "model_config": model.config(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "iteration": int(iteration),
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "parameter_count": count_parameters(model),
        "extra": extra or {},
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    _atomic_torch_save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: Optional[AlphaZeroNetwork] = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> Tuple[AlphaZeroNetwork, Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch signature
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid checkpoint: {path}")
    if model is None:
        model = AlphaZeroNetwork.from_config(payload.get("model_config", {}))
    model.load_state_dict(payload["model_state"], strict=strict)
    model.to(device)
    return model, payload


def load_model(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[AlphaZeroNetwork, Dict[str, Any]]:
    return load_checkpoint(path, model=None, device=device)


def create_initial_checkpoint(
    path: str | Path,
    config: dict,
    seed: int = 42,
    iteration: int = 0,
) -> Tuple[AlphaZeroNetwork, Path]:
    """Create a deterministic, randomly initialised model without pretrained weights."""
    torch.manual_seed(seed)
    model_config = config.get("model", config)
    model = AlphaZeroNetwork.from_config(model_config)
    destination = save_checkpoint(
        path,
        model,
        iteration=iteration,
        config=config,
        model_type="initial",
        extra={"seed": seed, "description": "Randomly initialized untrained network"},
    )
    return model, destination


def checkpoint_metadata(path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    model_config = payload.get("model_config", {})
    return {
        "path": str(path),
        "name": path.stem,
        "filename": path.name,
        "iteration": int(payload.get("iteration", 0)),
        "created_at": payload.get("created_at"),
        "created_at_iso": payload.get("created_at_iso"),
        "model_type": payload.get("model_type", _infer_type(path)),
        "parameter_count": payload.get("parameter_count"),
        "model_config": model_config,
        "config": payload.get("config", {}),
        "extra": payload.get("extra", {}),
        "size_bytes": path.stat().st_size,
    }


def _infer_type(path: Path) -> str:
    name = path.stem.lower()
    if name == "initial":
        return "initial"
    if name == "best":
        return "best"
    if name == "latest":
        return "latest"
    if name.startswith("iteration_"):
        return "iteration"
    return "checkpoint"


def list_checkpoints(directory: str | Path, recursive: bool = True) -> List[Dict[str, Any]]:
    directory = Path(directory)
    if not directory.exists():
        return []
    paths = directory.rglob("*.pt") if recursive else directory.glob("*.pt")
    records = []
    for path in paths:
        meta = checkpoint_metadata(path)
        if meta is not None:
            meta["experiment"] = _experiment_name(path, directory)
            records.append(meta)
    records.sort(key=lambda item: (item.get("iteration", 0), item.get("created_at") or 0), reverse=True)
    return records


def _experiment_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "default"
    except ValueError:
        return "unknown"


def find_named_checkpoint(directory: str | Path, name: str) -> Optional[Path]:
    directory = Path(directory)
    direct = directory / name
    if direct.exists():
        return direct
    matches = list(directory.rglob(name)) if directory.exists() else []
    return matches[0] if matches else None


def safe_promote_best(best_path: str | Path, candidate_model: AlphaZeroNetwork, **kwargs) -> Path:
    """Atomically replace best.pt only after the candidate saves and validates."""
    best_path = Path(best_path)
    backup = best_path.with_name("best_previous.pt")
    if best_path.exists():
        try:
            payload = torch.load(best_path, map_location="cpu", weights_only=False)
            torch.save(payload, backup)
        except Exception:
            pass
    return save_checkpoint(best_path, candidate_model, model_type="best", **kwargs)

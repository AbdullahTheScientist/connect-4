"""Experiment logging: text, JSONL metrics, status files and training control."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(log_file: Optional[str | Path] = None, name: str = "alphazero") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("timestamp", time.time())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
    tmp.replace(path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


class TrainingControl:
    """File-backed pause/stop protocol, readable by a GUI subprocess."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self.force_stop = False

    def request(self, paused: bool = False, stop: bool = False) -> None:
        if self.path is None:
            self.force_stop = stop
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, {"paused": paused, "stop": stop, "updated": time.time()})

    def clear(self) -> None:
        if self.path is None:
            self.force_stop = False
        else:
            self.request(False, False)

    def state(self) -> Dict[str, bool]:
        if self.path is not None:
            data = read_json(self.path, {}) or {}
            return {"paused": bool(data.get("paused")), "stop": bool(data.get("stop"))}
        return {"paused": False, "stop": self.force_stop}

    def wait_if_paused(self) -> None:
        announced = False
        while True:
            state = self.state()
            if state["stop"]:
                raise TrainingStopped("Training stop requested.")
            if not state["paused"]:
                return
            if not announced:
                print("Training paused. Waiting for resume...", flush=True)
                announced = True
            time.sleep(0.5)


class TrainingStopped(RuntimeError):
    pass


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np
        import torch

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.device):
            return str(obj)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

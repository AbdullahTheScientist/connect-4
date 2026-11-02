"""Evaluation metrics and historical record helpers."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    agent_a: str
    agent_b: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0
    total_moves: int = 0
    wins_as_first: int = 0
    wins_as_second: int = 0
    games_as_first: int = 0
    games_as_second: int = 0
    elapsed_seconds: float = 0.0
    simulations: int = 0
    iteration: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """Win=1, draw=0.5, loss=0 from agent A's perspective."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    @property
    def avg_game_length(self) -> float:
        return self.total_moves / self.games if self.games else 0.0

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "win_rate": self.win_rate,
                "loss_rate": self.loss_rate,
                "draw_rate": self.draw_rate,
                "score": self.score,
                "avg_game_length": self.avg_game_length,
            }
        )
        return data

    def summary(self) -> str:
        return (
            f"{self.agent_a} vs {self.agent_b}: {self.wins}W/{self.losses}L/{self.draws}D "
            f"(score {self.score:.1%}, avg length {self.avg_game_length:.1f})"
        )


def save_evaluation_result(
    result: EvaluationResult,
    path: str | Path,
    record_type: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = result.as_dict()
    if record_type:
        record["type"] = record_type
    if extra:
        record.update(extra)
    record["timestamp"] = time.time()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            if not isinstance(history, list):
                history = [history]
        except (json.JSONDecodeError, OSError):
            history = []
    else:
        history = []
    history.append(record)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    tmp.replace(path)
    return path


def load_evaluation_history(path: str | Path) -> list[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, OSError):
        return []


def aggregate_results(results: list[EvaluationResult], agent_a: str, agent_b: str) -> EvaluationResult:
    combined = EvaluationResult(agent_a=agent_a, agent_b=agent_b)
    for result in results:
        combined.wins += result.wins
        combined.losses += result.losses
        combined.draws += result.draws
        combined.games += result.games
        combined.total_moves += result.total_moves
        combined.wins_as_first += result.wins_as_first
        combined.wins_as_second += result.wins_as_second
        combined.games_as_first += result.games_as_first
        combined.games_as_second += result.games_as_second
        combined.elapsed_seconds += result.elapsed_seconds
    combined.simulations = results[0].simulations if results else 0
    return combined

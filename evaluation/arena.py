"""Evaluation arena for neural MCTS agents and real baselines."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np
import torch

from evaluation.baseline import RandomAgent, TacticalAgent
from evaluation.metrics import EvaluationResult
from game.connect4 import ConnectFour
from mcts.search import MCTS, SearchResult
from model.checkpoint import load_model
from model.network import AlphaZeroNetwork


class Agent(Protocol):
    name: str

    def select_action(self, game: ConnectFour, rng: Optional[np.random.Generator] = None) -> int: ...


class MCTSAgent:
    def __init__(
        self,
        model: AlphaZeroNetwork,
        config: dict,
        device: str | torch.device = "cpu",
        simulations: Optional[int] = None,
        name: str = "AlphaZero",
        temperature: float = 0.0,
        seed: int = 0,
        add_dirichlet_noise: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.simulations = int(
            simulations if simulations is not None else config.get("evaluation", {}).get("simulations", 50)
        )
        self.name = name
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)
        self.add_noise = add_dirichlet_noise
        self.last_result: Optional[SearchResult] = None
        self.model.eval()

    def select_action(self, game: ConnectFour, rng: Optional[np.random.Generator] = None) -> int:
        mcts_cfg = self.config.get("mcts", {})
        search = MCTS.from_model(
            self.model,
            device=self.device,
            simulations=self.simulations,
            c_puct=float(mcts_cfg.get("c_puct", 1.5)),
            dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(mcts_cfg.get("dirichlet_epsilon", 0.25)),
            rng=rng or self.rng,
        )
        result = search.search(game, add_dirichlet_noise=self.add_noise)
        self.last_result = result
        return result.select_action(temperature=self.temperature, rng=rng or self.rng)


@dataclass
class GameOutcome:
    winner: int
    length: int
    moves: List[int] = field(default_factory=list)
    agent_a_player: int = 1
    elapsed_seconds: float = 0.0

    @property
    def agent_a_won(self) -> bool:
        return self.winner == self.agent_a_player

    @property
    def agent_b_won(self) -> bool:
        return self.winner != 0 and self.winner != self.agent_a_player


class Arena:
    def __init__(self, config: dict) -> None:
        self.config = config

    def play_game(
        self,
        agent_a: Agent,
        agent_b: Agent,
        agent_a_player: int = 1,
        seed: int = 0,
    ) -> GameOutcome:
        game = ConnectFour()
        rng = np.random.default_rng(seed)
        moves: List[int] = []
        start = time.perf_counter()
        while not game.is_terminal():
            agent = agent_a if game.get_current_player() == agent_a_player else agent_b
            action = int(agent.select_action(game, rng=rng))
            if not game.is_legal(action):
                raise ValueError(f"Agent {agent.name} selected illegal action {action}")
            game.step(action)
            moves.append(action)
        return GameOutcome(
            winner=game.get_winner(),
            length=game.move_count,
            moves=moves,
            agent_a_player=agent_a_player,
            elapsed_seconds=time.perf_counter() - start,
        )

    def evaluate(
        self,
        agent_a: Agent,
        agent_b: Agent,
        games: int = 20,
        seed: int = 0,
        simulations: int = 0,
        iteration: Optional[int] = None,
        progress_callback=None,
    ) -> EvaluationResult:
        games = max(1, int(games))
        rng = np.random.default_rng(seed)
        result = EvaluationResult(
            agent_a=agent_a.name,
            agent_b=agent_b.name,
            games=games,
            simulations=simulations,
            iteration=iteration,
        )
        start = time.perf_counter()
        for index in range(games):
            # Alternate colours: agent A is player one on even games.
            a_player = 1 if index % 2 == 0 else 2
            game_seed = int(rng.integers(0, 2**31 - 1))
            outcome = self.play_game(agent_a, agent_b, agent_a_player=a_player, seed=game_seed)
            result.total_moves += outcome.length
            if a_player == 1:
                result.games_as_first += 1
            else:
                result.games_as_second += 1
            if outcome.winner == 0:
                result.draws += 1
            elif outcome.winner == a_player:
                result.wins += 1
                if a_player == 1:
                    result.wins_as_first += 1
                else:
                    result.wins_as_second += 1
            else:
                result.losses += 1
            if progress_callback:
                progress_callback(index + 1, games, result)
        result.elapsed_seconds = time.perf_counter() - start
        return result

    def evaluate_models(
        self,
        model_a: AlphaZeroNetwork,
        model_b: AlphaZeroNetwork,
        games: int,
        device: str | torch.device,
        simulations: int,
        name_a: str = "Candidate",
        name_b: str = "Best",
        seed: int = 0,
        iteration: Optional[int] = None,
        temperature: float = 0.0,
    ) -> EvaluationResult:
        agent_a = MCTSAgent(
            model_a,
            self.config,
            device=device,
            simulations=simulations,
            name=name_a,
            temperature=temperature,
            seed=seed,
        )
        agent_b = MCTSAgent(
            model_b,
            self.config,
            device=device,
            simulations=simulations,
            name=name_b,
            temperature=temperature,
            seed=seed + 1,
        )
        return self.evaluate(
            agent_a,
            agent_b,
            games=games,
            seed=seed,
            simulations=simulations,
            iteration=iteration,
        )

    def evaluate_against_random(
        self,
        model: AlphaZeroNetwork,
        games: int,
        device: str | torch.device,
        simulations: int,
        model_name: str = "Model",
        seed: int = 0,
        iteration: Optional[int] = None,
    ) -> EvaluationResult:
        agent = MCTSAgent(
            model,
            self.config,
            device=device,
            simulations=simulations,
            name=model_name,
            temperature=0.0,
            seed=seed,
        )
        random_agent = RandomAgent(seed=seed + 7)
        return self.evaluate(
            agent,
            random_agent,
            games=games,
            seed=seed,
            simulations=simulations,
            iteration=iteration,
        )

    def evaluate_against_tactical(
        self,
        model: AlphaZeroNetwork,
        games: int,
        device: str | torch.device,
        simulations: int,
        model_name: str = "Model",
        seed: int = 0,
        iteration: Optional[int] = None,
    ) -> EvaluationResult:
        agent = MCTSAgent(
            model,
            self.config,
            device=device,
            simulations=simulations,
            name=model_name,
            temperature=0.0,
            seed=seed,
        )
        tactical = TacticalAgent(seed=seed + 11)
        return self.evaluate(
            agent,
            tactical,
            games=games,
            seed=seed,
            simulations=simulations,
            iteration=iteration,
        )


def should_promote(result: EvaluationResult, threshold: float = 0.55) -> bool:
    """Promotion always comes from match results, never training loss."""
    return result.games > 0 and result.score >= float(threshold)


def agent_from_checkpoint(
    path: str | Path,
    config: dict,
    device: str | torch.device = "cpu",
    simulations: Optional[int] = None,
    name: Optional[str] = None,
    seed: int = 0,
) -> MCTSAgent:
    model, payload = load_model(path, device=device)
    return MCTSAgent(
        model,
        config,
        device=device,
        simulations=simulations,
        name=name or Path(path).stem,
        temperature=0.0,
        seed=seed,
    )


def build_agent(
    spec: str | Path,
    config: dict,
    device: str | torch.device = "cpu",
    simulations: Optional[int] = None,
    seed: int = 0,
) -> Agent:
    """Resolve 'random', 'tactical', or a checkpoint path to a real agent."""
    if isinstance(spec, str) and spec.lower() == "random":
        return RandomAgent(seed=seed)
    if isinstance(spec, str) and spec.lower() == "tactical":
        return TacticalAgent(seed=seed)
    return agent_from_checkpoint(
        spec,
        config,
        device=device,
        simulations=simulations,
        name=Path(path_name(spec)).stem,
        seed=seed,
    )


def path_name(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)

"""Self-play game generation with real MCTS and policy/value targets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch

from game.connect4 import ConnectFour
from mcts.search import MCTS
from model.network import AlphaZeroNetwork


@dataclass
class TrainingExample:
    observation: np.ndarray
    policy: np.ndarray
    player: int
    move_number: int
    value: float = 0.0

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, float]:
        return self.observation, self.policy, self.value


@dataclass
class GameRecord:
    examples: List[TrainingExample] = field(default_factory=list)
    winner: int = 0
    length: int = 0
    moves: List[int] = field(default_factory=list)
    seed: int = 0
    terminated: bool = False

    def finalize(self) -> "GameRecord":
        for example in self.examples:
            if self.winner == 0:
                example.value = 0.0
            elif self.winner == example.player:
                example.value = 1.0
            else:
                example.value = -1.0
        return self

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.examples:
            return (
                np.empty((0, 2, 6, 7), dtype=np.float16),
                np.empty((0, 7), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        observations = np.stack([e.observation for e in self.examples]).astype(np.float16)
        policies = np.stack([e.policy for e in self.examples]).astype(np.float32)
        values = np.asarray([e.value for e in self.examples], dtype=np.float32)
        return observations, policies, values


class SelfPlayGenerator:
    def __init__(
        self,
        model: AlphaZeroNetwork,
        config: dict,
        device: str | torch.device = "cpu",
        simulations: Optional[int] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.simulations = int(
            simulations
            if simulations is not None
            else config.get("mcts", {}).get("simulations", 50)
        )

    def _make_mcts(self, rng: np.random.Generator) -> MCTS:
        mcts_cfg = self.config.get("mcts", {})
        return MCTS.from_model(
            self.model,
            device=self.device,
            simulations=self.simulations,
            c_puct=float(mcts_cfg.get("c_puct", 1.5)),
            dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(mcts_cfg.get("dirichlet_epsilon", 0.25)),
            rng=rng,
        )

    def generate_game(
        self,
        seed: int = 0,
        add_noise: bool = True,
        temperature: Optional[float] = None,
        temperature_moves: Optional[int] = None,
    ) -> GameRecord:
        game = ConnectFour()
        rng = np.random.default_rng(seed)
        mcts = self._make_mcts(rng)
        selfplay_cfg = self.config.get("selfplay", {})
        temperature = float(
            temperature if temperature is not None else selfplay_cfg.get("temperature", 1.0)
        )
        temperature_moves = int(
            temperature_moves
            if temperature_moves is not None
            else selfplay_cfg.get("temperature_moves", 12)
        )
        record = GameRecord(seed=seed)
        self.model.eval()

        while not game.is_terminal():
            result = mcts.search(game, add_dirichlet_noise=add_noise)
            policy = result.action_probs.astype(np.float32).copy()
            if policy.sum() <= 0:
                mask = game.legal_action_mask()
                policy = mask / mask.sum()
            observation = game.get_observation()
            player = game.get_current_player()
            record.examples.append(
                TrainingExample(
                    observation=observation,
                    policy=policy,
                    player=player,
                    move_number=game.move_count,
                )
            )
            move_temperature = temperature if game.move_count < temperature_moves else 0.0
            action = result.select_action(temperature=move_temperature, rng=rng)
            if not game.is_legal(action):
                legal = game.get_legal_actions()
                action = int(rng.choice(legal))
            game.step(action)
            record.moves.append(int(action))

        record.winner = game.get_winner()
        record.length = game.move_count
        record.terminated = True
        return record.finalize()

    def generate_games(
        self,
        num_games: int,
        base_seed: int = 0,
        workers: int = 1,
        progress_callback: Optional[Callable[[int, int, GameRecord], None]] = None,
    ) -> List[GameRecord]:
        seeds = [base_seed + i for i in range(num_games)]
        if workers <= 1 or num_games <= 1:
            records = []
            for index, seed in enumerate(seeds):
                record = self.generate_game(seed)
                records.append(record)
                if progress_callback:
                    progress_callback(index + 1, num_games, record)
            return records

        records: List[Optional[GameRecord]] = [None] * num_games
        completed = 0
        with ThreadPoolExecutor(max_workers=min(workers, num_games)) as executor:
            futures = {executor.submit(self.generate_game, seed): i for i, seed in enumerate(seeds)}
            for future in as_completed(futures):
                index = futures[future]
                records[index] = future.result()
                completed += 1
                if progress_callback:
                    progress_callback(completed, num_games, records[index])
        return [record for record in records if record is not None]

    def generate_and_save(
        self,
        num_games: int,
        base_seed: int,
        output_path: str | Path,
        workers: int = 1,
        progress_callback: Optional[Callable[[int, int, GameRecord], None]] = None,
    ) -> tuple[List[GameRecord], Path]:
        records = self.generate_games(
            num_games,
            base_seed=base_seed,
            workers=workers,
            progress_callback=progress_callback,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_game_records(records, output_path)
        return records, output_path


def save_game_records(records: Iterable[GameRecord], path: str | Path) -> Path:
    path = Path(path)
    observations = []
    policies = []
    values = []
    players = []
    game_ids = []
    for game_id, record in enumerate(records):
        obs, pol, val = record.arrays()
        observations.append(obs)
        policies.append(pol)
        values.append(val)
        players.extend([e.player for e in record.examples])
        game_ids.extend([game_id] * len(record.examples))
    if observations and sum(len(x) for x in observations):
        all_obs = np.concatenate(observations)
        all_pol = np.concatenate(policies)
        all_val = np.concatenate(values)
    else:
        all_obs = np.empty((0, 2, 6, 7), dtype=np.float16)
        all_pol = np.empty((0, 7), dtype=np.float32)
        all_val = np.empty((0,), dtype=np.float32)
    np.savez(
        path,
        observations=all_obs,
        policies=all_pol,
        values=all_val,
        players=np.asarray(players, dtype=np.int8),
        game_ids=np.asarray(game_ids, dtype=np.int32),
        winners=np.asarray([r.winner for r in records], dtype=np.int8),
        lengths=np.asarray([r.length for r in records], dtype=np.int16),
        game_seeds=np.asarray([r.seed for r in records], dtype=np.int64),
    )
    return path

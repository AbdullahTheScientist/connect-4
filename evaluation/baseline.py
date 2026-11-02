"""Simple, real baselines: uniform random and a tactical heuristic agent."""
from __future__ import annotations

from typing import Optional

import numpy as np

from game.connect4 import ConnectFour


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, game: ConnectFour, rng: Optional[np.random.Generator] = None) -> int:
        legal = game.get_legal_actions()
        if len(legal) == 0:
            raise ValueError("No legal moves")
        generator = rng or self.rng
        return int(generator.choice(legal))


class TacticalAgent:
    """Immediate win, immediate block, then threat/center heuristic.

    This is not a neural network and is not trained. It provides a deterministic
    tactical opponent that is meaningfully stronger than uniform random play.
    """

    name = "Tactical"

    def __init__(self, seed: int = 0, randomness: float = 0.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.randomness = float(randomness)

    @staticmethod
    def _winning_moves_for_player(game: ConnectFour, player: int) -> list[int]:
        """Columns that would win immediately if `player` were to move."""
        moves: list[int] = []
        legal = game.get_legal_actions()
        for action in legal:
            probe = game.clone()
            if probe.is_terminal():
                continue
            probe.current_player = player
            probe.step(int(action))
            if probe.get_winner() == player:
                moves.append(int(action))
        return moves

    def select_action(self, game: ConnectFour, rng: Optional[np.random.Generator] = None) -> int:
        generator = rng or self.rng
        legal = [int(a) for a in game.get_legal_actions()]
        if not legal:
            raise ValueError("No legal moves")
        player = game.get_current_player()
        opponent = 3 - player

        immediate = self._winning_moves_for_player(game, player)
        if immediate:
            return int(immediate[0])

        blocks = self._winning_moves_for_player(game, opponent)
        if blocks:
            return int(blocks[0])

        if generator.random() < self.randomness:
            return int(generator.choice(legal))

        scored = []
        for action in legal:
            probe = game.clone()
            probe.step(action)
            if probe.is_terminal():
                score = 10_000.0
            else:
                opponent_threats = len(self._winning_moves_for_player(probe, opponent))
                own_future = len(self._winning_moves_for_player(probe, player))
                center_bonus = 3.0 - abs(3 - action)
                score = own_future * 50.0 + center_bonus - opponent_threats * 500.0
            scored.append((score, action))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top_score = scored[0][0]
        top = [action for score, action in scored if score == top_score]
        return int(generator.choice(top)) if len(top) > 1 else int(scored[0][1])

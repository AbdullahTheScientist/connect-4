"""A compact, dependency-light and fully tested Connect Four environment.

Board convention: numpy int8 array of shape (6, 7), row 0 is the top row.
Values: 0 = empty, 1 = player one, 2 = player two. Player one moves first.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


class ConnectFour:
    ROWS = 6
    COLS = 7
    CONNECT = 4
    EMPTY = 0
    PLAYER_ONE = 1
    PLAYER_TWO = 2

    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1) -> None:
        if board is None:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        else:
            board = np.asarray(board, dtype=np.int8)
            if board.shape != (self.ROWS, self.COLS):
                raise ValueError(f"Board must have shape {(self.ROWS, self.COLS)}, got {board.shape}")
            if not np.isin(board, (0, 1, 2)).all():
                raise ValueError("Board can only contain 0, 1 and 2")
            counts = [int((board == 1).sum()), int((board == 2).sum())]
            if abs(counts[0] - counts[1]) > 1:
                raise ValueError("Illegal piece counts for a Connect Four position")
            self.board = board.copy()
        if current_player not in (1, 2):
            raise ValueError("current_player must be 1 or 2")
        self.current_player = int(current_player)
        self.winner = 0
        self.terminal = False
        self.last_action: Optional[int] = None
        self.move_count = int((self.board != 0).sum())
        self._recompute_terminal()

    def _recompute_terminal(self) -> None:
        winner = self._find_winner()
        if winner:
            self.winner = winner
            self.terminal = True
        elif not np.any(self.board == 0):
            self.winner = 0
            self.terminal = True
        else:
            self.winner = 0
            self.terminal = False

    @classmethod
    def from_board(cls, board: np.ndarray, current_player: int) -> "ConnectFour":
        return cls(board=board, current_player=current_player)

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_player = 1
        self.winner = 0
        self.terminal = False
        self.last_action = None
        self.move_count = 0
        return self.get_observation()

    def clone(self) -> "ConnectFour":
        clone = ConnectFour.__new__(ConnectFour)
        clone.board = self.board.copy()
        clone.current_player = self.current_player
        clone.winner = self.winner
        clone.terminal = self.terminal
        clone.last_action = self.last_action
        clone.move_count = self.move_count
        return clone

    def get_legal_actions(self) -> np.ndarray:
        if self.terminal:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.board[0] == 0).astype(np.int64)

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.COLS, dtype=np.float32)
        if not self.terminal:
            mask[self.board[0] == 0] = 1.0
        return mask

    def is_legal(self, action: int) -> bool:
        return (
            not self.terminal
            and isinstance(action, (int, np.integer))
            and 0 <= int(action) < self.COLS
            and self.board[0, int(action)] == 0
        )

    def _drop_row(self, action: int) -> int:
        column = self.board[:, action]
        empty = np.flatnonzero(column == 0)
        if len(empty) == 0:
            raise ValueError(f"Column {action} is full")
        return int(empty[-1])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.terminal:
            raise RuntimeError("Cannot step a terminal game. Call reset().")
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        if not self.is_legal(int(action)):
            raise ValueError(f"Illegal action: {action}")

        action = int(action)
        row = self._drop_row(action)
        player = self.current_player
        self.board[row, action] = player
        self.last_action = action
        self.move_count += 1
        self.current_player = 3 - player

        if self._would_win_from(row, action, player):
            self.winner = player
            self.terminal = True
            reward = 1.0
        elif self.move_count == self.ROWS * self.COLS:
            self.winner = 0
            self.terminal = True
            reward = 0.0
        else:
            reward = 0.0

        info: Dict[str, object] = {
            "player": player,
            "row": row,
            "winner": self.winner,
            "done": self.terminal,
        }
        return self.get_observation(), reward, self.terminal, info

    def _find_winner(self) -> int:
        for player in (1, 2):
            for row in range(self.ROWS):
                for col in range(self.COLS):
                    if self.board[row, col] != player:
                        continue
                    if self._would_win_from(row, col, player):
                        return player
        return 0

    def _would_win_from(self, row: int, col: int, player: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.CONNECT:
                return True
        return False

    def get_observation(self) -> np.ndarray:
        """Two float32 planes from the player-to-move's perspective."""
        own = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == (3 - self.current_player)).astype(np.float32)
        return np.stack([own, opponent], axis=0)

    def get_current_player(self) -> int:
        return self.current_player

    def is_terminal(self) -> bool:
        return self.terminal

    def get_winner(self) -> int:
        """Return 1 or 2 for a winner, 0 for no winner/draw."""
        return self.winner

    def get_winning_line(self) -> list[tuple[int, int]]:
        """Return the four (or more) cells forming a winning line, if any."""
        if not self.terminal or self.winner == 0:
            return []
        player = self.winner
        for row in range(self.ROWS):
            for col in range(self.COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(row + i * dr, col + i * dc) for i in range(self.CONNECT)]
                    if all(
                        0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player
                        for r, c in cells
                    ):
                        return cells
        return []

    def get_result(self, player: int) -> float:
        if not self.terminal:
            raise RuntimeError("Game is not finished")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def is_full(self) -> bool:
        return not np.any(self.board == 0)

    def render_ascii(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [" " + " ".join(str(i) for i in range(self.COLS))]
        for row in self.board:
            rows.append("|" + "|".join(symbols[int(v)] for v in row) + "|")
        rows.append("+" + "-" * (self.COLS * 2 - 1) + "+")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"ConnectFour(current_player={self.current_player}, terminal={self.terminal}, "
            f"winner={self.winner}, moves={self.move_count})"
        )

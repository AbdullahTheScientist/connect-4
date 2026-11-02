"""MCTS tree node."""
from __future__ import annotations

from typing import Dict, Optional

from game.connect4 import ConnectFour


class Node:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "action",
        "parent",
        "children",
        "player",
        "state",
    )

    def __init__(
        self,
        state: ConnectFour,
        prior: float = 1.0,
        action: Optional[int] = None,
        parent: Optional["Node"] = None,
    ) -> None:
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.action = action
        self.parent = parent
        self.children: Dict[int, "Node"] = {}
        self.player = state.get_current_player()
        self.state = state

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def is_expanded(self) -> bool:
        return bool(self.children) or self.state.is_terminal()

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += float(value)

    def add_child(self, action: int, child: "Node") -> None:
        self.children[int(action)] = child

    def __repr__(self) -> str:
        return (
            f"Node(action={self.action}, player={self.player}, prior={self.prior:.3f}, "
            f"visits={self.visit_count}, value={self.mean_value:.3f})"
        )

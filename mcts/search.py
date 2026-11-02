"""PUCT Monte Carlo Tree Search with neural network leaf evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch

from game.connect4 import ConnectFour
from mcts.node import Node
from model.network import AlphaZeroNetwork


Evaluator = Callable[[np.ndarray], tuple[np.ndarray, float]]


class NeuralEvaluator:
    """Thread-safe inference wrapper around a policy-value network."""

    def __init__(self, model: AlphaZeroNetwork, device: str | torch.device = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)

    def __call__(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        tensor = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(tensor)
        return policy[0].detach().cpu().numpy().astype(np.float64), float(value[0].detach().cpu().item())


@dataclass
class SearchResult:
    action_probs: np.ndarray
    visit_counts: np.ndarray
    root_value: float
    simulations: int
    legal_actions: np.ndarray
    root: Optional[Node] = field(default=None, repr=False)
    nn_prior: np.ndarray = field(default_factory=lambda: np.zeros(7))

    def policy(self, temperature: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        counts = self.visit_counts.astype(np.float64)
        if counts.sum() <= 0:
            legal = self.legal_actions
            policy = np.zeros(7, dtype=np.float64)
            if len(legal):
                policy[legal] = 1.0 / len(legal)
            return policy
        if temperature <= 1e-8:
            policy = np.zeros(7, dtype=np.float64)
            best = np.flatnonzero(counts == counts.max())
            chosen = int(best[0]) if rng is None else int(rng.choice(best))
            policy[chosen] = 1.0
            return policy
        powered = np.power(counts, 1.0 / max(temperature, 1e-3))
        policy = powered / powered.sum()
        return policy

    def select_action(
        self,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        policy = self.policy(temperature=temperature, rng=rng)
        if temperature <= 1e-8:
            return int(np.argmax(policy))
        if rng is None:
            return int(np.argmax(policy))
        return int(rng.choice(np.arange(7), p=policy))


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.evaluator = evaluator
        self.simulations = max(1, int(simulations))
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_epsilon = float(dirichlet_epsilon)
        self.rng = rng or np.random.default_rng()

    @classmethod
    def from_model(
        cls,
        model: AlphaZeroNetwork,
        device: str | torch.device = "cpu",
        simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        rng: Optional[np.random.Generator] = None,
    ) -> "MCTS":
        return cls(
            NeuralEvaluator(model, device),
            simulations=simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            rng=rng,
        )

    def terminal_value(self, node: Node) -> float:
        if not node.state.is_terminal():
            return 0.0
        winner = node.state.get_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == node.player else -1.0

    def search(
        self,
        game: ConnectFour,
        add_dirichlet_noise: bool = False,
        simulations: Optional[int] = None,
    ) -> SearchResult:
        root = Node(game.clone())
        legal = game.get_legal_actions()
        if game.is_terminal() or len(legal) == 0:
            return SearchResult(
                action_probs=np.zeros(7, dtype=np.float64),
                visit_counts=np.zeros(7, dtype=np.int32),
                root_value=self._terminal_value(game),
                simulations=0,
                legal_actions=legal,
                root=root,
            )

        noise_pending = bool(add_dirichlet_noise)
        n_sims = self.simulations if simulations is None else max(1, int(simulations))
        for _ in range(n_sims):
            self._simulate(root, apply_root_noise=noise_pending)
            noise_pending = False

        counts = np.zeros(7, dtype=np.int32)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        total = counts.sum()
        probs = counts.astype(np.float64) / total if total else game.legal_action_mask() / max(1, len(legal))
        nn_prior = np.zeros(7, dtype=np.float64)
        for action, child in root.children.items():
            nn_prior[action] = child.prior
        return SearchResult(
            action_probs=probs,
            visit_counts=counts,
            root_value=root.mean_value,
            simulations=n_sims,
            legal_actions=legal,
            root=root,
            nn_prior=nn_prior,
        )

    def _simulate(self, root: Node, apply_root_noise: bool = False) -> None:
        node = root
        path = [node]

        while node.children:
            node = self._select_child(node)
            path.append(node)

        if node.state.is_terminal():
            value = self.terminal_value(node)
        else:
            value = self._expand(node)
            if apply_root_noise and node is root and node.children:
                self._add_dirichlet_noise(root)

        leaf_player = node.player
        for ancestor in reversed(path):
            ancestor_value = value if ancestor.player == leaf_player else -value
            ancestor.update(ancestor_value)

    def _expand(self, node: Node) -> float:
        observation = node.state.get_observation()
        policy, value = self.evaluator(observation)
        policy = np.asarray(policy, dtype=np.float64).reshape(-1)
        legal = node.state.get_legal_actions()
        masked = np.zeros(7, dtype=np.float64)
        if len(legal):
            masked[legal] = np.maximum(policy[legal], 0.0)
            if masked.sum() <= 0:
                masked[legal] = 1.0
            masked /= masked.sum()
        for action in legal:
            child_state = node.state.clone()
            child_state.step(int(action))
            child = Node(
                state=child_state,
                prior=float(masked[action]),
                action=int(action),
                parent=node,
            )
            node.add_child(int(action), child)
        return float(np.clip(value, -1.0, 1.0))

    def _select_child(self, node: Node) -> Node:
        best_score = -float("inf")
        best_child = None
        sqrt_parent_visits = math.sqrt(max(1, node.visit_count))
        for action in sorted(node.children):
            child = node.children[action]
            # Child values are from the child player's perspective; negate for parent.
            q_value = -child.mean_value
            exploration = (
                self.c_puct
                * child.prior
                * sqrt_parent_visits
                / (1.0 + child.visit_count)
            )
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Cannot select from a node without children")
        return best_child

    def _add_dirichlet_noise(self, root: Node) -> None:
        if not root.children:
            return
        actions = sorted(root.children)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, eta in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(eta)
        total = sum(child.prior for child in root.children.values())
        if total > 0:
            for child in root.children.values():
                child.prior /= total

    def get_action(
        self,
        game: ConnectFour,
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
    ) -> tuple[int, SearchResult]:
        result = self.search(game, add_dirichlet_noise=add_dirichlet_noise)
        if game.is_terminal():
            raise RuntimeError("No action available in a terminal game")
        action = result.select_action(temperature=temperature, rng=self.rng)
        return action, result

    @staticmethod
    def _terminal_value(game: ConnectFour) -> float:
        if not game.is_terminal():
            return 0.0
        winner = game.get_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == game.get_current_player() else -1.0


class CachingEvaluator:
    """Optional evaluator wrapper for repeated UI queries."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.cache: Dict[bytes, tuple[np.ndarray, float]] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, observation: np.ndarray) -> tuple[np.ndarray, float]:
        key = np.asarray(observation, dtype=np.float32).tobytes()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        result = self.evaluator(observation)
        self.cache[key] = result
        return result

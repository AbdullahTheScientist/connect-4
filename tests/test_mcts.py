import numpy as np
import pytest
import torch

from game.connect4 import ConnectFour
from mcts.node import Node
from mcts.search import MCTS, NeuralEvaluator
from model.network import AlphaZeroNetwork


def uniform_evaluator(observation):
    return np.ones(7, dtype=np.float64) / 7.0, 0.0


def play_sequence(actions):
    game = ConnectFour()
    for action in actions:
        game.step(action)
    return game


def test_terminal_node_values():
    node = Node(ConnectFour())
    assert node.visit_count == 0
    assert node.mean_value == 0.0
    node.update(1.0)
    assert node.mean_value == 1.0


def test_search_returns_legal_distribution():
    game = ConnectFour()
    search = MCTS(uniform_evaluator, simulations=24, rng=np.random.default_rng(1))
    result = search.search(game)
    assert result.simulations == 24
    assert result.visit_counts.shape == (7,)
    assert np.isclose(result.action_probs.sum(), 1.0)
    assert all(result.action_probs[a] >= 0 for a in game.get_legal_actions())
    assert result.visit_counts.sum() == 23  # first simulation expands the root


def test_search_excludes_full_columns():
    game = ConnectFour()
    for _ in range(6):
        game.step(3)
    # It is player two's turn and column three is full.
    result = MCTS(uniform_evaluator, simulations=20, rng=np.random.default_rng(3)).search(game)
    assert result.visit_counts[3] == 0
    assert result.action_probs[3] == 0


def test_terminal_search_has_no_actions():
    game = play_sequence([0, 1, 0, 1, 0, 1, 0])
    result = MCTS(uniform_evaluator, simulations=10).search(game)
    assert result.simulations == 0
    assert result.root_value == -1.0
    assert result.visit_counts.sum() == 0


def test_mcts_finds_immediate_win():
    # Player one has three in column zero; player two has three in column one.
    game = play_sequence([0, 1, 0, 1, 0, 1])
    result = MCTS(uniform_evaluator, simulations=50, rng=np.random.default_rng(4)).search(game)
    action = result.select_action(temperature=0.0)
    assert action == 0


def test_terminal_win_child_has_positive_q():
    game = play_sequence([0, 1, 0, 1, 0, 1])
    result = MCTS(uniform_evaluator, simulations=30, rng=np.random.default_rng(5)).search(game)
    child = result.root.children[0]
    assert child.state.is_terminal()
    assert child.state.get_winner() == 1
    # Child player is 2 (the loser), so the parent values this child positively.
    assert -child.mean_value > 0


def test_temperature_zero_is_greedy():
    game = ConnectFour()
    result = MCTS(uniform_evaluator, simulations=12, rng=np.random.default_rng(0)).search(game)
    policy = result.policy(temperature=0.0)
    assert policy.max() == 1.0
    assert policy.sum() == 1.0
    assert result.select_action(0.0) == int(np.argmax(result.visit_counts))


def test_temperature_one_matches_visit_distribution():
    game = ConnectFour()
    result = MCTS(uniform_evaluator, simulations=15, rng=np.random.default_rng(2)).search(game)
    expected = result.visit_counts / result.visit_counts.sum()
    assert np.allclose(result.policy(temperature=1.0), expected)


def test_root_noise_changes_priors_but_stays_normalized():
    game = ConnectFour()
    mcts = MCTS(uniform_evaluator, simulations=6, rng=np.random.default_rng(8))
    result = mcts.search(game, add_dirichlet_noise=True)
    priors = np.array([child.prior for child in result.root.children.values()])
    assert np.isclose(priors.sum(), 1.0)
    assert np.isclose(result.action_probs.sum(), 1.0)


def test_neural_mcts_with_random_network():
    torch.manual_seed(0)
    model = AlphaZeroNetwork(channels=8, residual_blocks=1)
    game = ConnectFour()
    result = MCTS.from_model(model, simulations=8, rng=np.random.default_rng(0)).search(game)
    assert result.visit_counts.sum() == 7  # first simulation expands the root
    assert np.isclose(result.action_probs.sum(), 1.0)
    action = result.select_action(temperature=0.0)
    assert game.is_legal(action)


def test_neural_evaluator_returns_valid_outputs():
    model = AlphaZeroNetwork(channels=8, residual_blocks=1)
    evaluator = NeuralEvaluator(model)
    policy, value = evaluator(ConnectFour().get_observation())
    assert policy.shape == (7,)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert -1.0 <= value <= 1.0


def test_empty_game_search_never_selects_illegal():
    game = ConnectFour()
    game.step(0)
    mcts = MCTS(uniform_evaluator, simulations=10, rng=np.random.default_rng(9))
    for _ in range(10):
        action = mcts.get_action(game, temperature=1.0)[0]
        assert game.is_legal(action)

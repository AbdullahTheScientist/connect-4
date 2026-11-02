import json

import numpy as np

from evaluation.arena import Arena, MCTSAgent, should_promote
from evaluation.baseline import RandomAgent, TacticalAgent
from evaluation.metrics import EvaluationResult, save_evaluation_result
from model.checkpoint import save_checkpoint
from model.network import AlphaZeroNetwork
from utils.config import load_config


def test_metrics_properties():
    result = EvaluationResult("A", "B", wins=6, losses=3, draws=1, games=10, total_moves=250)
    assert np.isclose(result.win_rate, 0.6)
    assert np.isclose(result.loss_rate, 0.3)
    assert np.isclose(result.draw_rate, 0.1)
    assert np.isclose(result.score, 0.65)
    assert np.isclose(result.avg_game_length, 25.0)


def test_save_evaluation_result_preserves_record_type(tmp_path):
    path = tmp_path / "evaluations.json"
    result = EvaluationResult("candidate", "best", wins=12, losses=6, draws=2, games=20)
    save_evaluation_result(
        result,
        path,
        record_type="candidate_vs_best",
        extra={"iteration": 7, "promoted": True},
    )
    save_evaluation_result(result, path, record_type="best_vs_random")

    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    assert len(history) == 2
    assert history[0]["type"] == "candidate_vs_best"
    assert history[0]["iteration"] == 7
    assert history[0]["promoted"] is True
    assert history[1]["type"] == "best_vs_random"
    assert history[1]["games"] == 20


def test_promotion_threshold():
    result = EvaluationResult("candidate", "best", wins=11, losses=7, draws=2, games=20)
    assert should_promote(result, 0.6)
    assert not should_promote(result, 0.7)
    empty = EvaluationResult("candidate", "best")
    assert not should_promote(empty)


def test_tactical_beats_random():
    config = load_config()
    arena = Arena(config)
    result = arena.evaluate(TacticalAgent(seed=0), RandomAgent(seed=1), games=12, seed=2)
    assert result.games == 12
    assert result.wins + result.losses + result.draws == 12
    assert result.score > 0.6


def test_arena_alternates_colours():
    class FirstLegalAgent:
        def __init__(self, name):
            self.name = name

        def select_action(self, game, rng=None):
            return int(game.get_legal_actions()[0])

    config = load_config()
    arena = Arena(config)
    result = arena.evaluate(FirstLegalAgent("A"), FirstLegalAgent("B"), games=6, seed=1)
    assert result.games_as_first == 3
    assert result.games_as_second == 3
    assert result.games == 6


def test_neural_agent_evaluation(tmp_path):
    config = load_config()
    config["model"] = {"channels": 8, "residual_blocks": 1}
    config["mcts"]["simulations"] = 2
    model_a = AlphaZeroNetwork.from_config(config["model"])
    model_b = AlphaZeroNetwork.from_config(config["model"])
    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    save_checkpoint(path_a, model_a)
    save_checkpoint(path_b, model_b)
    agent_a = MCTSAgent(model_a, config, simulations=2, name="A", seed=0)
    agent_b = MCTSAgent(model_b, config, simulations=2, name="B", seed=1)
    arena = Arena(config)
    result = arena.evaluate(agent_a, agent_b, games=2, seed=4, simulations=2)
    assert result.games == 2
    assert result.wins + result.losses + result.draws == 2
    assert result.avg_game_length > 0


def test_agent_selects_legal_action():
    config = load_config()
    config["model"] = {"channels": 8, "residual_blocks": 1}
    model = AlphaZeroNetwork.from_config(config["model"])
    from game.connect4 import ConnectFour

    game = ConnectFour()
    agent = MCTSAgent(model, config, simulations=4, name="test", seed=0)
    action = agent.select_action(game)
    assert game.is_legal(action)
    assert agent.last_result is not None

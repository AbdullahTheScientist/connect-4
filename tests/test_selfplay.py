import numpy as np
import pytest
import torch

from selfplay.generator import SelfPlayGenerator, save_game_records
from model.network import AlphaZeroNetwork
from utils.config import load_config


@pytest.fixture
def config():
    cfg = load_config()
    cfg["model"] = {"channels": 8, "residual_blocks": 1, "input_channels": 2}
    cfg["mcts"]["simulations"] = 2
    cfg["mcts"]["dirichlet_epsilon"] = 0.1
    cfg["selfplay"]["temperature"] = 1.0
    cfg["selfplay"]["temperature_moves"] = 3
    return cfg


def test_selfplay_game_completes_with_valid_targets(config):
    torch.manual_seed(0)
    model = AlphaZeroNetwork.from_config(config["model"])
    generator = SelfPlayGenerator(model, config, device="cpu")
    record = generator.generate_game(seed=1, add_noise=False)
    assert record.terminated
    assert record.length == len(record.moves)
    assert len(record.examples) == record.length
    assert 0 <= len(record.moves) <= 42
    for example in record.examples:
        assert example.observation.shape == (2, 6, 7)
        assert example.policy.shape == (7,)
        assert np.isclose(example.policy.sum(), 1.0, atol=1e-5)
        if record.winner == 0:
            assert example.value == 0.0
        elif record.winner == example.player:
            assert example.value == 1.0
        else:
            assert example.value == -1.0
    players = [e.player for e in record.examples]
    assert players[0] == 1
    assert all(a != b for a, b in zip(players, players[1:]))


def test_selfplay_parallel_generation(config):
    model = AlphaZeroNetwork.from_config(config["model"])
    generator = SelfPlayGenerator(model, config, device="cpu")
    records = generator.generate_games(3, base_seed=100, workers=2)
    assert len(records) == 3
    assert all(record.terminated for record in records)
    assert len({record.seed for record in records}) == 3


def test_training_examples_persistence(config, tmp_path):
    model = AlphaZeroNetwork.from_config(config["model"])
    generator = SelfPlayGenerator(model, config, device="cpu")
    records = generator.generate_games(2, base_seed=5)
    path = save_game_records(records, tmp_path / "games.npz")
    assert path.exists()
    with np.load(path) as data:
        assert data["observations"].shape[1:] == (2, 6, 7)
        assert data["policies"].shape[1] == 7
        assert data["observations"].dtype == np.float16
        assert len(data["values"]) == sum(len(r.examples) for r in records)


def test_policy_targets_are_not_one_hot_always(config):
    model = AlphaZeroNetwork.from_config(config["model"])
    generator = SelfPlayGenerator(model, config, device="cpu", simulations=6)
    record = generator.generate_game(seed=4)
    nonzero_counts = [int((e.policy > 0).sum()) for e in record.examples]
    assert max(nonzero_counts) > 1

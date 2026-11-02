import json
from pathlib import Path

import numpy as np
import torch

from training.losses import policy_value_loss
from training.trainer import Trainer
from utils.config import load_config


def tiny_config(tmp_path: Path) -> dict:
    config = load_config()
    config["experiment"] = {"name": "test_train", "seed": 7, "deterministic": True}
    config["model"] = {"channels": 8, "residual_blocks": 1, "input_channels": 2}
    config["mcts"]["simulations"] = 2
    config["selfplay"].update({"games_per_iteration": 2, "workers": 1, "temperature_moves": 2, "save_examples": True})
    config["training"].update(
        {
            "iterations": 1,
            "epochs": 1,
            "batch_size": 8,
            "max_batches_per_epoch": 2,
            "min_buffer_size": 4,
            "replay_capacity": 400,
            "lr_step_size": 1,
            "save_buffer": True,
        }
    )
    config["evaluation"].update(
        {
            "candidate_games": 2,
            "random_games": 2,
            "tactical_games": 0,
            "simulations": 2,
            "promotion_threshold": 0.5,
        }
    )
    config["paths"] = {
        "checkpoints": str(tmp_path / "checkpoints"),
        "logs": str(tmp_path / "logs"),
        "results": str(tmp_path / "results"),
    }
    config["tensorboard"]["enabled"] = False
    return config


def test_loss_components():
    policy = torch.tensor([[0.7, 0.3], [0.2, 0.8]], requires_grad=True)
    value = torch.tensor([0.5, -0.5], requires_grad=True)
    target_policy = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_value = torch.tensor([1.0, -1.0])
    total, policy_loss, value_loss, l2 = policy_value_loss(
        policy, value, target_policy, target_value, l2_coefficient=0.0
    )
    expected_policy = -(np.log(0.7) + np.log(0.8)) / 2
    expected_value = ((0.5 - 1.0) ** 2 + (-0.5 + 1.0) ** 2) / 2
    assert np.isclose(policy_loss.detach().item(), expected_policy, atol=1e-6)
    assert np.isclose(value_loss.detach().item(), expected_value, atol=1e-6)
    assert l2.detach().item() == 0.0
    assert np.isclose(total.detach().item(), expected_policy + expected_value, atol=1e-6)
    total.backward()
    assert policy.grad is not None
    assert value.grad is not None


def test_l2_regularization_term():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
    policy = torch.tensor([[0.5, 0.5]])
    value = torch.tensor([0.0])
    _, _, _, l2 = policy_value_loss(
        policy,
        value,
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([0.0]),
        parameters=[parameter],
        l2_coefficient=0.5,
    )
    assert np.isclose(l2.detach().item(), 2.5, atol=1e-6)


def test_one_iteration_pipeline_and_resume(tmp_path):
    config = tiny_config(tmp_path)
    trainer = Trainer(config, experiment="test_train", device="cpu")
    metrics = trainer.train()
    assert len(metrics) == 1
    record = metrics[0]
    assert record["iteration"] == 1
    assert record["buffer_size"] > 0
    assert np.isfinite(record["policy_loss"])
    assert np.isfinite(record["value_loss"])
    assert "candidate_score_vs_best" in record

    checkpoint_dir = Path(config["paths"]["checkpoints"]) / "test_train"
    for name in ("initial.pt", "latest.pt", "best.pt", "iteration_0001.pt", "replay_buffer.npz"):
        assert (checkpoint_dir / name).exists(), name
    with (checkpoint_dir / "initial.pt").open("rb"):
        pass
    logs = Path(config["paths"]["logs"]) / "test_train" / "metrics.jsonl"
    assert logs.exists()
    records = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    assert (Path(config["paths"]["results"]) / "test_train" / "evaluations.json").exists()
    trainer.close()

    resumed = Trainer(config, experiment="test_train", resume=checkpoint_dir / "latest.pt", device="cpu")
    assert resumed.iteration == 1
    assert len(resumed.buffer) > 0
    resumed.close()


def test_training_requires_valid_resume_path(tmp_path):
    import pytest

    config = tiny_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        Trainer(config, experiment="test_train", resume=tmp_path / "missing.pt", device="cpu")

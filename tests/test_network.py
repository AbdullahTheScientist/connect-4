import numpy as np
import pytest
import torch

from model.checkpoint import (
    checkpoint_metadata,
    create_initial_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from model.network import AlphaZeroNetwork, count_parameters, summarize_model


def test_network_output_shapes_and_ranges():
    model = AlphaZeroNetwork(channels=16, residual_blocks=2)
    x = torch.randn(4, 2, 6, 7)
    policy, value = model(x)
    assert policy.shape == (4, 7)
    assert value.shape == (4,)
    assert torch.allclose(policy.sum(dim=1), torch.ones(4), atol=1e-5)
    assert (policy >= 0).all()
    assert (value >= -1).all() and (value <= 1).all()


def test_network_gradients_flow():
    model = AlphaZeroNetwork(channels=8, residual_blocks=1)
    policy, value = model(torch.randn(2, 2, 6, 7))
    loss = policy.sum() + value.sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_parameter_counting():
    model = AlphaZeroNetwork(channels=16, residual_blocks=1)
    count = count_parameters(model)
    assert count > 1_000
    assert summarize_model(model)["parameters"] == count


def test_checkpoint_roundtrip_and_optimizer(tmp_path):
    model = AlphaZeroNetwork(channels=8, residual_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    policy, value = model(torch.randn(1, 2, 6, 7))
    (policy.sum() + value.sum()).backward()
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer=optimizer, iteration=7, config={"model": model.config()})
    loaded, payload = load_checkpoint(path)
    assert payload["iteration"] == 7
    assert "optimizer_state" in payload
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])
    meta = checkpoint_metadata(path)
    assert meta["iteration"] == 7
    assert meta["parameter_count"] == count_parameters(model)


def test_initial_checkpoint_is_deterministic(tmp_path):
    config = {"model": {"channels": 8, "residual_blocks": 1}}
    model_a, path_a = create_initial_checkpoint(tmp_path / "initial.pt", config, seed=123)
    reference = {k: v.clone() for k, v in model_a.state_dict().items()}
    model_b, path_b = create_initial_checkpoint(tmp_path / "initial2.pt", config, seed=123)
    for key, value in model_b.state_dict().items():
        assert torch.equal(value, reference[key])
    assert path_a.exists() and path_b.exists()


def test_checkpoint_discovery(tmp_path):
    config = {"model": {"channels": 8, "residual_blocks": 1}}
    for name in ("best.pt", "latest.pt", "iteration_0002.pt"):
        model = AlphaZeroNetwork.from_config(config["model"])
        save_checkpoint(tmp_path / name, model, iteration=2)
    records = list_checkpoints(tmp_path)
    names = {r["filename"] for r in records}
    assert names == {"best.pt", "latest.pt", "iteration_0002.pt"}


def test_invalid_input_dimensions():
    model = AlphaZeroNetwork(channels=8, residual_blocks=1)
    with pytest.raises(ValueError):
        model(torch.randn(2, 6, 7))

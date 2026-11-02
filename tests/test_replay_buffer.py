import numpy as np
import pytest

from training.replay_buffer import ReplayBuffer


def make_data(n):
    observations = np.random.rand(n, 2, 6, 7).astype(np.float32)
    policies = np.random.rand(n, 7).astype(np.float32)
    policies /= policies.sum(axis=1, keepdims=True)
    values = np.random.uniform(-1, 1, size=n).astype(np.float32)
    return observations, policies, values


def test_add_and_sample():
    buffer = ReplayBuffer(capacity=100)
    obs, pol, val = make_data(20)
    buffer.add(obs, pol, val)
    assert len(buffer) == 20
    b_obs, b_pol, b_val = buffer.sample(8, rng=np.random.default_rng(0))
    assert b_obs.shape == (8, 2, 6, 7)
    assert b_obs.dtype == np.float32
    assert b_pol.shape == (8, 7)
    assert b_val.shape == (8,)


def test_sample_without_replacement():
    buffer = ReplayBuffer(capacity=50)
    buffer.add(*make_data(30))
    _, _, values = buffer.sample(30, rng=np.random.default_rng(4))
    assert len(np.unique(values)) == 30


def test_capacity_and_order():
    buffer = ReplayBuffer(capacity=5)
    obs, pol, val = make_data(8)
    buffer.add(obs, pol, val)
    assert len(buffer) == 5
    snapshot = buffer.snapshot()
    assert np.allclose(snapshot["values"], val[3:])


def test_multi_add_with_wrap():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(*make_data(7))
    buffer.add(*make_data(6))
    assert len(buffer) == 10
    assert buffer.position == 3


def test_oversized_add_keeps_latest():
    buffer = ReplayBuffer(capacity=10)
    obs, pol, val = make_data(25)
    buffer.add(obs, pol, val)
    assert np.allclose(buffer.snapshot()["values"], val[-10:])


def test_save_and_load(tmp_path):
    buffer = ReplayBuffer(capacity=100)
    buffer.add(*make_data(17))
    path = buffer.save(tmp_path / "buffer.npz")
    loaded = ReplayBuffer.load(path)
    assert len(loaded) == 17
    assert loaded.capacity >= 100
    original = buffer.snapshot()
    restored = loaded.snapshot()
    assert np.allclose(original["values"], restored["values"])


def test_empty_buffer_errors():
    buffer = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buffer.sample(4)


def test_invalid_shapes():
    buffer = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError):
        buffer.add(np.zeros((2, 2, 5, 7)), np.ones((2, 7)) / 7, np.zeros(2))
    with pytest.raises(ValueError):
        buffer.add(np.zeros((2, 2, 6, 7)), np.ones((2, 6)) / 6, np.zeros(2))

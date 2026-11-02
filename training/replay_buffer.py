"""Fixed-capacity circular replay buffer with memory-efficient storage."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 50_000,
        observation_shape: tuple[int, ...] = (2, 6, 7),
        policy_size: int = 7,
        dtype: np.dtype = np.float16,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)
        self.policy_size = int(policy_size)
        self.dtype = np.dtype(dtype)
        self.observations = np.empty((capacity, *self.observation_shape), dtype=self.dtype)
        self.policies = np.empty((capacity, self.policy_size), dtype=np.float32)
        self.values = np.empty((capacity,), dtype=np.float32)
        self.size = 0
        self.position = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observations: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        observations = np.asarray(observations)
        policies = np.asarray(policies, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        n = len(values)
        if n == 0:
            return
        if observations.shape[1:] != self.observation_shape:
            raise ValueError(f"Expected observations shape (*, {self.observation_shape}), got {observations.shape}")
        if policies.shape != (n, self.policy_size):
            raise ValueError(f"Expected policies shape ({n}, {self.policy_size}), got {policies.shape}")
        if len(observations) != n:
            raise ValueError("Observations and values have mismatched lengths")
        if not np.isfinite(values).all():
            raise ValueError("Values contain NaN or infinity")

        if n >= self.capacity:
            observations = observations[-self.capacity :]
            policies = policies[-self.capacity :]
            values = values[-self.capacity :]
            n = self.capacity
            self.observations[:] = observations.astype(self.dtype)
            self.policies[:] = policies
            self.values[:] = values
            self.size = self.capacity
            self.position = 0
            self.total_added += n
            return

        end = self.position + n
        if end <= self.capacity:
            self.observations[self.position : end] = observations.astype(self.dtype)
            self.policies[self.position : end] = policies
            self.values[self.position : end] = values
        else:
            first = self.capacity - self.position
            self.observations[self.position :] = observations[:first].astype(self.dtype)
            self.policies[self.position :] = policies[:first]
            self.values[self.position :] = values[:first]
            remaining = n - first
            self.observations[:remaining] = observations[first:].astype(self.dtype)
            self.policies[:remaining] = policies[first:]
            self.values[:remaining] = values[first:]
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_added += n

    def add_examples(self, observations: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        self.add(observations, policies, values)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_size = min(int(batch_size), self.size)
        rng = rng or np.random.default_rng()
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.observations[indices].astype(np.float32),
            self.policies[indices].copy(),
            self.values[indices].copy(),
        )

    def snapshot(self) -> dict:
        size = self.size
        if size < self.capacity:
            return {
                "observations": self.observations[:size].copy(),
                "policies": self.policies[:size].copy(),
                "values": self.values[:size].copy(),
                "position": self.position,
                "total_added": self.total_added,
                "capacity": self.capacity,
            }
        indices = np.concatenate((np.arange(self.position, self.capacity), np.arange(0, self.position)))
        return {
            "observations": self.observations[indices].copy(),
            "policies": self.policies[indices].copy(),
            "values": self.values[indices].copy(),
            "position": self.position,
            "total_added": self.total_added,
            "capacity": self.capacity,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        np.savez(
            path,
            observations=data["observations"],
            policies=data["policies"],
            values=data["values"],
            position=np.asarray(data["position"]),
            total_added=np.asarray(data["total_added"]),
            capacity=np.asarray(data["capacity"]),
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        capacity: Optional[int] = None,
        observation_shape: tuple[int, ...] = (2, 6, 7),
    ) -> "ReplayBuffer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as data:
            observations = data["observations"]
            policies = data["policies"]
            values = data["values"]
            saved_position = int(data.get("position", 0))
            total_added = int(data.get("total_added", len(values)))
            saved_capacity = int(data.get("capacity", len(values)))
        target_capacity = max(int(capacity or saved_capacity or len(values) or 1), len(values))
        buffer = cls(target_capacity, observation_shape=observation_shape, policy_size=policies.shape[1])
        buffer.add(observations, policies, values)
        buffer.position = saved_position % target_capacity if target_capacity else 0
        buffer.total_added = total_added
        return buffer

    def clear(self) -> None:
        self.size = 0
        self.position = 0
        self.total_added = 0

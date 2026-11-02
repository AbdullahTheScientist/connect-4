"""Policy-value residual convolutional network for Connect Four."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class AlphaZeroNetwork(nn.Module):
    """Small AlphaZero-style two-headed residual network.

    Input: (B, 2, 6, 7), planes from the current player's perspective.
    Output: (policy probabilities (B, 7), value (B,) in [-1, 1]).
    """

    def __init__(
        self,
        channels: int = 64,
        residual_blocks: int = 4,
        input_channels: int = 2,
        policy_size: int = 7,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.residual_blocks = int(residual_blocks)
        self.input_channels = int(input_channels)
        self.policy_size = int(policy_size)
        self.input_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(residual_blocks)])
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, policy_size)
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.policy_fc.weight, std=0.01)
        nn.init.normal_(self.value_fc2.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected 4-D input (B, C, H, W), got shape {tuple(x.shape)}")
        x = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        x = self.tower(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        policy = self.policy_fc(policy.flatten(1))
        policy = F.softmax(policy, dim=1)

        value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        value = F.relu(self.value_fc1(value.flatten(1)), inplace=True)
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)
        return policy, value

    def config(self) -> dict:
        return {
            "channels": self.channels,
            "residual_blocks": self.residual_blocks,
            "input_channels": self.input_channels,
            "policy_size": self.policy_size,
        }

    @classmethod
    def from_config(cls, config: dict) -> "AlphaZeroNetwork":
        return cls(
            channels=int(config.get("channels", 64)),
            residual_blocks=int(config.get("residual_blocks", 4)),
            input_channels=int(config.get("input_channels", 2)),
            policy_size=int(config.get("policy_size", 7)),
        )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def summarize_model(model: nn.Module) -> dict:
    return {
        "parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        **model.config(),
    }

"""AlphaZero losses with explicit L2 regularization."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def policy_value_loss(
    policy_logits_or_probs: torch.Tensor,
    value_pred: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter] | None = None,
    l2_coefficient: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute policy CE, value MSE, L2 and total loss.

    Network outputs are probabilities, so log() is safe with a small clamp.
    """
    log_policy = torch.log(policy_logits_or_probs.clamp_min(1e-8))
    policy_loss = -(policy_target * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(value_pred.reshape(-1), value_target.reshape(-1))
    if parameters is not None and l2_coefficient > 0:
        l2 = torch.zeros((), device=value_pred.device, dtype=value_pred.dtype)
        for parameter in parameters:
            if parameter.requires_grad and parameter.ndim > 1:
                l2 = l2 + parameter.pow(2).sum()
        l2 = l2 * l2_coefficient
    else:
        l2 = torch.zeros((), device=value_pred.device, dtype=value_pred.dtype)
    total = policy_loss + value_loss + l2
    return total, policy_loss, value_loss, l2

"""
Pinball loss for quantile regression.

Paper Section 11.2: L_quantile = Σ_τ [τ * max(y - qτ, 0) + (1-τ) * max(qτ - y, 0)]
"""

import torch
import torch.nn as nn
from torch import Tensor


class PinballLoss(nn.Module):
    """Pinball (quantile) loss for multi-quantile prediction."""

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: Tensor, targets: Tensor, masks: Tensor) -> Tensor:
        """
        Args:
            predictions: (batch, n_quantiles)
            targets: (batch, 1)
            masks: (batch, 1)
        Returns:
            scalar loss
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        n_valid = masks.sum().clamp(min=1.0)

        for i, tau in enumerate(self.quantiles):
            q_pred = predictions[:, i:i+1]
            error = targets - q_pred
            loss = torch.max(tau * error, (tau - 1) * error)
            total_loss = total_loss + (loss * masks).sum() / n_valid

        return total_loss / len(self.quantiles)

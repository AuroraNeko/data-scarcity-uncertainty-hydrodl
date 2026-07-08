"""
physics_loss.py  -  Physics-guided loss functions for LPU-Stream.

Paper reference (Sections 10.3-10.4):
    1. Non-negative penalty: L_nonneg = mean(max(0, -Q_hat))
    2. Extreme event weighting: w = 1 + alpha * I(Q_obs > Q95)
    3. Rainfall-runoff monotonicity: L_mono = mean(max(0, Q(X) - Q(X+delta)))
    4. Annual water balance: L_wb = |sum(Q_hat) - sum(Q_obs)| / sum(Q_obs) per basin-year

Total: L = weighted_MSE + lambda_nonneg * L_nonneg + lambda_mono * L_mono + lambda_wb * L_wb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PhysicsLoss(nn.Module):
    """Combined physics-guided loss for LPU-Stream training."""

    def __init__(
        self,
        q95: float,
        prcp_std: float = 7.58,
        delta_raw: float = 2.0,
        lambda_nonneg: float = 0.05,
        lambda_mono: float = 0.1,
        lambda_wb: float = 0.1,
        extreme_alpha: float = 2.0,
        use_nonneg: bool = True,
        use_mono: bool = True,
        use_wb: bool = True,
        use_extreme: bool = True,
    ):
        super().__init__()
        self.q95 = q95
        self.delta_norm = delta_raw / prcp_std
        self.lambda_nonneg = lambda_nonneg
        self.lambda_mono = lambda_mono
        self.lambda_wb = lambda_wb
        self.extreme_alpha = extreme_alpha
        self.use_nonneg = use_nonneg
        self.use_mono = use_mono
        self.use_wb = use_wb
        self.use_extreme = use_extreme

    def compute_extreme_weights(self, target: Tensor, mask: Tensor) -> Tensor:
        """Compute sample weights: 1 + alpha * I(target > Q95)."""
        if not self.use_extreme:
            return torch.ones_like(target)
        is_extreme = (target > self.q95).float()
        return 1.0 + self.extreme_alpha * is_extreme

    def compute_nonneg_loss(self, predictions: Tensor, masks: Tensor) -> Tensor:
        """L_nonneg = mean(max(0, -Q_hat))  -  soft penalty for negative predictions."""
        if not self.use_nonneg or self.lambda_nonneg == 0:
            return torch.tensor(0.0, device=predictions.device)
        neg_violation = F.relu(-predictions)
        return (neg_violation * masks).sum() / masks.sum().clamp(min=1.0)

    def compute_monotonicity_loss(
        self, model: nn.Module, dynamic_seq: Tensor,
        static_attrs: Tensor, pred_original: Tensor,
    ) -> Tensor:
        """L_mono = mean(max(0, Q(X) - Q(X + delta))) over valid samples."""
        if not self.use_mono or self.lambda_mono == 0:
            return torch.tensor(0.0, device=dynamic_seq.device)

        # Augment precipitation (channel 0) by delta
        dynamic_aug = dynamic_seq.clone()
        dynamic_aug[:, :, 0] = dynamic_aug[:, :, 0] + self.delta_norm

        with torch.no_grad():
            pred_augmented = model(dynamic_aug, static_attrs)

        # Penalize: predicted flow should not decrease when precip increases
        violation = F.relu(pred_original - pred_augmented)
        return violation.mean()

    def compute_water_balance_loss(
        self, predictions: Tensor, targets: Tensor,
        masks: Tensor, basin_indices: Tensor, year_indices: Tensor,
    ) -> Tensor:
        """L_wb = mean over basin-years of |sum(Q_hat) - sum(Q_obs)| / sum(Q_obs).

        Vectorized implementation using scatter_add for efficiency.
        """
        if not self.use_wb or self.lambda_wb == 0:
            return torch.tensor(0.0, device=predictions.device)

        pred = predictions.squeeze()
        tgt = targets.squeeze()
        msk = masks.squeeze()
        bids = basin_indices.long().squeeze()
        yids = year_indices.long().squeeze()

        # Valid samples only
        valid = msk > 0
        if valid.sum() < 10:
            return torch.tensor(0.0, device=predictions.device)

        pred_v = pred[valid]
        tgt_v = tgt[valid]

        # Composite key: basin * 1000 + year
        composite = bids[valid] * 1000 + yids[valid]

        # Remap to contiguous indices for scatter
        unique_keys, inverse = composite.unique(return_inverse=True)
        n_groups = unique_keys.size(0)

        if n_groups == 0:
            return torch.tensor(0.0, device=predictions.device)

        # Accumulate sums per group
        sum_pred = torch.zeros(n_groups, device=pred.device)
        sum_tgt = torch.zeros(n_groups, device=pred.device)
        sum_pred.scatter_add_(0, inverse, pred_v)
        sum_tgt.scatter_add_(0, inverse, tgt_v)

        # Filter: enough samples and non-trivial target sum
        valid_groups = sum_tgt.abs() > 1.0
        if valid_groups.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        wb_ratio = (sum_pred[valid_groups] - sum_tgt[valid_groups]).abs() / sum_tgt[valid_groups].abs()
        return wb_ratio.mean()

    def forward(
        self, model: nn.Module, dynamic_seq: Tensor, static_attrs: Tensor,
        predictions: Tensor, targets: Tensor, masks: Tensor,
        basin_indices: Tensor, year_indices: Tensor,
    ) -> dict[str, Tensor]:
        """Compute all loss components and return dict."""
        # Weighted MSE
        weights = self.compute_extreme_weights(targets, masks)
        mse_per_sample = F.mse_loss(predictions, targets, reduction="none")
        weighted_mse = (mse_per_sample * weights * masks).sum() / masks.sum().clamp(min=1.0)

        # Non-negative penalty
        nonneg_loss = self.compute_nonneg_loss(predictions, masks)

        # Monotonicity
        mono_loss = self.compute_monotonicity_loss(model, dynamic_seq, static_attrs, predictions)

        # Water balance
        wb_loss = self.compute_water_balance_loss(
            predictions, targets, masks, basin_indices, year_indices,
        )

        # Total
        total = (weighted_mse
                 + self.lambda_nonneg * nonneg_loss
                 + self.lambda_mono * mono_loss
                 + self.lambda_wb * wb_loss)

        return {
            "total": total,
            "mse": weighted_mse.detach(),
            "nonneg": nonneg_loss.detach(),
            "mono": mono_loss.detach(),
            "wb": wb_loss.detach(),
        }

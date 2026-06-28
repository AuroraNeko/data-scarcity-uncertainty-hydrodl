"""
TCN (Temporal Convolutional Network) for streamflow prediction.

Reference: Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling"

Architecture (paper Section 8.5):
    Causal dilated Conv1D -> residual blocks -> prediction head
    Static attributes concatenated at the end.
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal convolution with left-side padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Single TCN residual block with two dilated causal convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection (1x1 conv if channel size changes)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        res = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """TCN for streamflow prediction with static attribute fusion."""

    def __init__(
        self,
        n_dynamic: int = 5,
        n_static: int = 13,
        channels: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Build TCN layers with exponentially increasing dilation
        layers = []
        in_ch = n_dynamic
        for i in range(n_blocks):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, channels, kernel_size, dilation, dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

        # Prediction head: TCN output (last timestep) + static attributes
        self.head = nn.Sequential(
            nn.Linear(channels + n_static, channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, dynamic_seq, static_attrs):
        """
        Args:
            dynamic_seq: (batch, seq_len, n_dynamic)
            static_attrs: (batch, n_static)
        Returns:
            prediction: (batch, 1)
        """
        # TCN expects (batch, channels, seq_len)
        x = dynamic_seq.permute(0, 2, 1)
        x = self.tcn(x)
        # Take last timestep
        x = x[:, :, -1]
        x = self.dropout(x)

        combined = torch.cat([x, static_attrs], dim=-1)
        return self.head(combined)

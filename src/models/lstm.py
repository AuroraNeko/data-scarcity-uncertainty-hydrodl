"""
LSTM baseline model for CAMELS-US streamflow prediction.

Architecture used by the point baseline and endpoint sensitivity:
    dynamic sequence -> LSTM -> hidden state
    static attributes -> concatenate with hidden state -> point or quantile head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """Standard LSTM with static-at-output fusion and point/quantile heads."""

    def __init__(
        self,
        n_dynamic: int = 15,
        n_static: int = 13,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        quantiles: list[float] | None = None,
        enforce_monotonic: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.quantiles = quantiles
        self.n_outputs = len(quantiles) if quantiles else 1
        self.enforce_monotonic = enforce_monotonic

        self.lstm = nn.LSTM(
            input_size=n_dynamic,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Prediction head: LSTM hidden + static attributes -> streamflow
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.n_outputs),
        )

    def forward(self, dynamic_seq, static_attrs):
        """
        Args:
            dynamic_seq: (batch, seq_len, n_dynamic)
            static_attrs: (batch, n_static)
        Returns:
            prediction: (batch, 1) in point mode or (batch, n_quantiles)
        """
        lstm_out, (h_n, _) = self.lstm(dynamic_seq)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)

        # Concatenate with static attributes
        combined = torch.cat([last_hidden, static_attrs], dim=-1)
        prediction = self.head(combined)
        if self.enforce_monotonic and self.quantiles is not None:
            if self.n_outputs != 3 or self.quantiles != [0.05, 0.50, 0.95]:
                raise ValueError(
                    "Monotonic parameterization requires ordered [0.05, 0.50, 0.95] quantiles"
                )
            median = prediction[:, 1:2]
            lower = median - F.softplus(prediction[:, 0:1])
            upper = median + F.softplus(prediction[:, 2:3])
            prediction = torch.cat([lower, median, upper], dim=1)
        return prediction

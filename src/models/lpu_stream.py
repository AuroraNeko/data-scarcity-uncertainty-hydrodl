"""Compact regional recurrent model with point and quantile output heads.

Architecture:
    Static attributes -> MLP embedding -> z_basin (32-dim)
    z_basin concatenated to each timestep of dynamic input
    -> nn.LSTM (cuDNN accelerated)
    -> concat(LSTM hidden, z_basin) -> prediction head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StaticCatchmentEmbedding(nn.Module):
    """Map static catchment attributes to a learned basin embedding."""

    def __init__(
        self,
        n_static: int = 13,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x_static: Tensor) -> Tensor:
        return self.net(x_static)


class LPUStreamModel(nn.Module):
    """LSTM backbone with a basin embedding at each input time step.

    Dropout note: with a single LSTM layer the LSTM's internal dropout is a
    no-op, so dropout (default p=0.3) is applied to the final hidden state and
    inside the prediction head instead.
    """

    def __init__(
        self,
        n_dynamic: int = 15,
        n_static: int = 13,
        hidden_size: int = 128,
        embed_dim: int = 32,
        dropout: float = 0.3,
        quantiles: list[float] | None = None,
        enforce_monotonic: bool = False,
        **kwargs,  # saved model_config dicts carry extra descriptive keys
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.quantiles = quantiles  # None = point prediction, list = quantile mode
        self.n_outputs = len(quantiles) if quantiles else 1
        self.enforce_monotonic = enforce_monotonic
        if self.enforce_monotonic:
            if (
                self.quantiles is None
                or len(self.quantiles) != 3
                or self.quantiles[1] != 0.5
                or not self.quantiles[0] < self.quantiles[1] < self.quantiles[2]
            ):
                raise ValueError(
                    "Monotonic parameterization requires three strictly ordered "
                    "quantiles with 0.5 in the center"
                )

        # Static catchment embedding
        self.embedding = StaticCatchmentEmbedding(n_static, embed_dim)

        # LSTM receives dynamic features + basin embedding at each timestep
        self.lstm = nn.LSTM(
            input_size=n_dynamic + embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_size + embed_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.n_outputs),
        )

    def forward(self, dynamic_seq: Tensor, static_attrs: Tensor) -> Tensor:
        """
        Args:
            dynamic_seq: (batch, seq_len, n_dynamic)
            static_attrs: (batch, n_static)
        Returns:
            point mode: (batch, 1)
            quantile mode: (batch, n_quantiles) e.g. [Q_0.05, Q_0.5, Q_0.95]
        """
        z_basin = self.embedding(static_attrs)  # (batch, embed_dim)
        z_expanded = z_basin.unsqueeze(1).expand(-1, dynamic_seq.size(1), -1)
        lstm_input = torch.cat([dynamic_seq, z_expanded], dim=-1)

        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        h_final = h_n[-1]  # (batch, hidden_size)
        h_final = self.dropout(h_final)

        combined = torch.cat([h_final, z_basin], dim=-1)
        output = self.head(combined)
        if self.enforce_monotonic and self.quantiles is not None:
            median = output[:, 1:2]
            lower = median - F.softplus(output[:, 0:1])
            upper = median + F.softplus(output[:, 2:3])
            output = torch.cat([lower, median, upper], dim=1)
        return output

"""
LPU-Stream: Lightweight Physics-guided Uncertainty-calibrated Streamflow forecasting.

Paper reference (Sections 9.3-9.5):
    Static catchment embedding + LSTM temporal encoder + prediction head.

Architecture:
    Static attributes → MLP embedding → z_basin (32-dim)
    z_basin concatenated to each timestep of dynamic input
    → nn.LSTM (cuDNN accelerated)
    → concat(LSTM hidden, z_basin) → prediction head
"""

import torch
import torch.nn as nn
from torch import Tensor


class StaticCatchmentEmbedding(nn.Module):
    """MLP-based catchment attribute embedding.

    Paper Section 9.3: Linear(13,64) → ReLU → Dropout → Linear(64,32)
    """

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
    """LPU-Stream core model for streamflow prediction.

    Paper Section 9.4-9.5: LSTM backbone with basin embedding
    concatenated to each timestep input.

    Dropout note: with a single LSTM layer the LSTM's internal dropout is a
    no-op, so dropout (default p=0.3) is applied to the final hidden state and
    inside the prediction head instead.
    """

    def __init__(
        self,
        n_dynamic: int = 5,
        n_static: int = 13,
        hidden_size: int = 128,
        embed_dim: int = 32,
        dropout: float = 0.3,
        quantiles: list[float] | None = None,
        no_static: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_dim = 0 if no_static else embed_dim
        self.quantiles = quantiles  # None = point prediction, list = quantile mode
        self.n_outputs = len(quantiles) if quantiles else 1
        self.no_static = no_static

        # Static catchment embedding
        if not no_static:
            self.embedding = StaticCatchmentEmbedding(n_static, embed_dim)
        else:
            self.embedding = None

        # LSTM receives dynamic features + basin embedding at each timestep
        self.lstm = nn.LSTM(
            input_size=n_dynamic if no_static else n_dynamic + embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        # Prediction head
        head_input = hidden_size if no_static else hidden_size + embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_size // 2),
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
        # Basin embedding (skip if no_static)
        if self.no_static:
            z_basin = None
            lstm_input = dynamic_seq
        else:
            z_basin = self.embedding(static_attrs)  # (batch, embed_dim)
            z_expanded = z_basin.unsqueeze(1).expand(-1, dynamic_seq.size(1), -1)
            lstm_input = torch.cat([dynamic_seq, z_expanded], dim=-1)

        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        h_final = h_n[-1]  # (batch, hidden_size)
        h_final = self.dropout(h_final)

        # Combine LSTM output with basin embedding (if available)
        if self.no_static:
            combined = h_final
        else:
            combined = torch.cat([h_final, z_basin], dim=-1)
        return self.head(combined)

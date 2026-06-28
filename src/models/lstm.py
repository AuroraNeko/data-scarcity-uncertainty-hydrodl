"""
LSTM baseline model for CAMELS-US streamflow prediction.

Architecture (paper Section 8.3):
    dynamic sequence -> LSTM -> hidden state
    static attributes -> concatenate with hidden state -> Linear head -> prediction
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """Standard LSTM for streamflow prediction with static attribute fusion."""

    def __init__(
        self,
        n_dynamic: int = 5,
        n_static: int = 13,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

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
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, dynamic_seq, static_attrs):
        """
        Args:
            dynamic_seq: (batch, seq_len, n_dynamic)
            static_attrs: (batch, n_static)
        Returns:
            prediction: (batch, 1)
        """
        lstm_out, (h_n, _) = self.lstm(dynamic_seq)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)

        # Concatenate with static attributes
        combined = torch.cat([last_hidden, static_attrs], dim=-1)
        prediction = self.head(combined)
        return prediction

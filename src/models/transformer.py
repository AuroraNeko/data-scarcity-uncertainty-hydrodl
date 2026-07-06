"""
Small Transformer encoder for streamflow prediction.

Paper reference (Section 8.6):
    Encoder-only, 2 layers, hidden 128, 4 heads

Provides a modern DL baseline to show LPU-Stream can match
Transformer performance with a simpler architecture + physics constraints.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 400, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Small Transformer encoder for streamflow prediction."""

    def __init__(
        self,
        n_dynamic: int = 15,
        n_static: int = 13,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(n_dynamic, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head: last timestep output + static attributes
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_static, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, dynamic_seq, static_attrs):
        """
        Args:
            dynamic_seq: (batch, seq_len, n_dynamic)
            static_attrs: (batch, n_static)
        Returns:
            prediction: (batch, 1)
        """
        x = self.input_proj(dynamic_seq)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Use last timestep
        x = x[:, -1]
        x = self.dropout(x)

        combined = torch.cat([x, static_attrs], dim=-1)
        return self.head(combined)

"""
EA-LSTM (Entity-Aware LSTM) for streamflow prediction.

Reference: Kratzert et al. (2019) "Towards learning universal, global
hydrological models from large-sample datasets"

Key idea: static catchment attributes generate the input gate of LSTM,
allowing the model to learn basin-specific behavior.

Architecture:
    static attributes -> Input Gate Generator (Linear) -> input gate
    dynamic sequence -> modified LSTM (with static-controlled gate)
    last hidden state -> prediction head -> streamflow
"""

import torch
import torch.nn as nn
from torch import Tensor


class EALSTMCell(nn.Module):
    """LSTM cell where the input gate is generated from static attributes.

    Holds the ``gate_linear`` and ``dynamic_linear`` parameters used by the
    JIT-compiled :func:`_ea_lstm_loop`. The ``forward`` method below is kept
    for reference but is **not** called during inference — ``EALSTMModel``
    runs the fused ``_ea_lstm_loop`` for speed. The cell is instantiated so
    its sub-modules register as parameters of the parent model.
    """

    def __init__(self, n_dynamic: int, n_static: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Input gate from static attributes (the EA part)
        self.gate_linear = nn.Linear(n_static, hidden_size)

        # Standard LSTM gates (forget, output, cell candidate) from dynamic input + hidden
        self.dynamic_linear = nn.Linear(n_dynamic + hidden_size, 3 * hidden_size)

    def forward(self, x_dynamic: Tensor, h: Tensor, c: Tensor, x_static: Tensor):
        i = torch.sigmoid(self.gate_linear(x_static))
        combined = torch.cat([x_dynamic, h], dim=-1)
        f, o, g = torch.split(self.dynamic_linear(combined), self.hidden_size, dim=-1)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


@torch.jit.script
def _ea_lstm_loop(
    dynamic_seq: Tensor,
    static_attrs: Tensor,
    gate_weight: Tensor,
    gate_bias: Tensor,
    dyn_weight: Tensor,
    dyn_bias: Tensor,
    hidden_size: int,
) -> Tensor:
    """JIT-compiled EA-LSTM sequential loop — runs as fused CUDA kernel."""
    batch_size = dynamic_seq.size(0)
    h = torch.zeros(batch_size, hidden_size, device=dynamic_seq.device, dtype=dynamic_seq.dtype)
    c = torch.zeros(batch_size, hidden_size, device=dynamic_seq.device, dtype=dynamic_seq.dtype)

    # Pre-compute input gate (static, same for all timesteps)
    i = torch.sigmoid(torch.nn.functional.linear(static_attrs, gate_weight, gate_bias))

    for t in range(dynamic_seq.size(1)):
        combined = torch.cat([dynamic_seq[:, t], h], dim=-1)
        gates = torch.nn.functional.linear(combined, dyn_weight, dyn_bias)
        f, o, g = gates.split(hidden_size, dim=-1)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)

    return h


class EALSTMModel(nn.Module):
    """EA-LSTM for streamflow prediction with entity-aware input gating."""

    def __init__(
        self,
        n_dynamic: int = 5,
        n_static: int = 13,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell = EALSTMCell(n_dynamic, n_static, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, dynamic_seq: Tensor, static_attrs: Tensor) -> Tensor:
        h = _ea_lstm_loop(
            dynamic_seq, static_attrs,
            self.cell.gate_linear.weight, self.cell.gate_linear.bias,
            self.cell.dynamic_linear.weight, self.cell.dynamic_linear.bias,
            self.hidden_size,
        )
        h = self.dropout(h)
        return self.head(h)

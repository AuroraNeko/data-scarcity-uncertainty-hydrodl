"""Custom-head LPU-Stream variants for the full-671 baselines.

All variants reuse the base LPU-Stream backbone unchanged (static
embedding at every time step, 128-unit LSTM, 64-unit head) and replace only
the output width and the loss/quantile semantics:

MDNStream   K-component Gaussian mixture head (Klotz et al. 2022, "GMM" style)
GaussianQuantileStream  Gaussian location-scale head mu + sigma * Phi^-1(tau),
            trained with pinball loss on freshly drawn taus. This is not UMAL.
NoiseStream mu + sigma head trained with Gaussian NLL (MC Dropout + aleatoric)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.lpu_stream import LPUStreamModel  # noqa: E402

SQRT2 = 1.4142135623730951


class ChunkedForwardMixin:
    """Sequence-chunked, gradient-checkpointed LSTM pass.

    A single cuDNN call on the protocol shape (batch 1024, 365 steps,
    hidden 128) reserves ~9 GB of workspace on this 8 GB card; under WDDM
    the excess spills to system RAM and training throttles to ~0.6-1.4 s
    per batch as soon as any desktop context competes. Splitting the
    sequence into 73-step chunks that carry (h, c) forward reproduces the
    same recurrence exactly (cuDNN's implicit zero initial state is
    supplied explicitly) while keeping reserved VRAM under ~2 GB, which
    benchmarks at ~81 ms per training batch with ~5 GB left for the
    desktop. The chunks contain no dropout or RNG, so activation
    recomputation in backward is deterministic.
    """

    CHUNK = 73

    def _chunked_lstm(self, lstm_input: Tensor):
        h = lstm_input.new_zeros(1, lstm_input.shape[0], self.hidden_size)
        c = lstm_input.new_zeros(1, lstm_input.shape[0], self.hidden_size)

        def run(chunk, hh, cc):
            _out, (ho, co) = self.lstm(chunk, (hh, cc))
            return ho, co

        for s in range(0, lstm_input.shape[1], self.CHUNK):
            h, c = torch.utils.checkpoint.checkpoint(
                run, lstm_input[:, s:s + self.CHUNK], h, c,
                use_reentrant=False)
        return h

    def forward(self, dynamic_seq: Tensor, static_attrs: Tensor) -> Tensor:
        z_basin = self.embedding(static_attrs)  # (batch, embed_dim)
        z_expanded = z_basin.unsqueeze(1).expand(-1, dynamic_seq.size(1), -1)
        lstm_input = torch.cat([dynamic_seq, z_expanded], dim=-1)

        h_final = self.dropout(self._chunked_lstm(lstm_input)[-1])

        combined = torch.cat([h_final, z_basin], dim=-1)
        output = self.head(combined)
        if self.enforce_monotonic and self.quantiles is not None:
            median = output[:, 1:2]
            lower = median - F.softplus(output[:, 0:1])
            upper = median + F.softplus(output[:, 2:3])
            output = torch.cat([lower, median, upper], dim=1)
        return output


class PointChunkedStream(ChunkedForwardMixin, LPUStreamModel):
    """Point-output variant used by the reselection run."""


class ChunkedPrimary(ChunkedForwardMixin, LPUStreamModel):
    """Equivalent carried-state recurrence for memory-safe 365-day fitting."""


class HeadStreamModel(ChunkedForwardMixin, LPUStreamModel):
    """Published backbone with a custom output width; head rebuilt verbatim."""

    def __init__(self, n_out: int, dropout: float = 0.3, **kwargs):
        super().__init__(quantiles=None, dropout=dropout, **kwargs)
        self.n_outputs = n_out
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size + self.embed_dim, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, n_out),
        )


# --------------------------------------------------------------------- MDN --
class MDNStream(HeadStreamModel):
    K = 5

    def __init__(self, **kwargs):
        super().__init__(n_out=3 * self.K, **kwargs)

    def split(self, raw: Tensor):
        logits_pi = raw[:, : self.K]
        mu = raw[:, self.K: 2 * self.K]
        sigma = F.softplus(raw[:, 2 * self.K: 3 * self.K]) + 1e-4
        return logits_pi, mu, sigma

    def nll(self, raw: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        logits_pi, mu, sigma = self.split(raw)
        logw = torch.log_softmax(logits_pi, dim=-1)
        log_prob = (
            -0.5 * ((target - mu) / sigma) ** 2
            - torch.log(sigma)
            - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        )
        loss = -torch.logsumexp(logw + log_prob, dim=-1, keepdim=True)
        n_valid = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / n_valid

    def predict_quantiles(self, raw: Tensor, taus=(0.05, 0.5, 0.95), iters=80) -> Tensor:
        logits_pi, mu, sigma = self.split(raw)
        w = torch.softmax(logits_pi, dim=-1)
        lo = (mu - 12 * sigma).min(dim=-1).values - 5.0
        hi = (mu + 12 * sigma).max(dim=-1).values + 5.0
        outputs = []
        for tau in taus:
            a, b = lo.clone(), hi.clone()
            for _ in range(iters):
                mid = 0.5 * (a + b)
                z = (mid.unsqueeze(-1) - mu) / sigma
                cdf = (w * 0.5 * (1 + torch.erf(z / SQRT2))).sum(dim=-1)
                low = cdf < tau
                a = torch.where(low, mid, a)
                b = torch.where(low, b, mid)
            outputs.append(0.5 * (a + b))
        return torch.stack(outputs, dim=-1)


# ----------------------------------------- Gaussian location-scale quantiles --
class GaussianQuantileStream(HeadStreamModel):
    """Gaussian location-scale distribution trained by random-quantile pinball."""

    N_TAU = 8

    def __init__(self, **kwargs):
        super().__init__(n_out=2, **kwargs)

    def split(self, raw: Tensor):
        mu = raw[:, 0:1]
        sigma = F.softplus(raw[:, 1:2]) + 1e-4
        return mu, sigma

    def pinball(self, raw: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        mu, sigma = self.split(raw)
        taus = torch.rand(target.shape[0], self.N_TAU, device=target.device) * 0.98 + 0.01
        z = torch.special.ndtri(taus).unsqueeze(-1)  # (batch, n_tau, 1)
        q = mu.unsqueeze(1) + sigma.unsqueeze(1) * z
        error = target.unsqueeze(1) - q
        loss = torch.max(taus.unsqueeze(-1) * error, (taus.unsqueeze(-1) - 1) * error)
        n_valid = mask.sum().clamp(min=1.0)
        # Mean over taus and keep the result (batch, 1): one extra dimension
        # would broadcast against the (batch, 1) mask into (batch, batch, 1)
        # and inflate the loss by ~batch_size.
        return (loss.mean(dim=1) * mask).sum() / n_valid

    def predict_quantiles(self, raw: Tensor, taus=(0.05, 0.5, 0.95)) -> Tensor:
        mu, sigma = self.split(raw)
        z = torch.special.ndtri(torch.tensor(taus, device=raw.device))
        return mu + sigma * z.unsqueeze(0)


# --------------------------------------------------------- MC + noise head --
class NoiseStream(HeadStreamModel):
    def __init__(self, **kwargs):
        super().__init__(n_out=2, **kwargs)

    def split(self, raw: Tensor):
        mu = raw[:, 0:1]
        sigma = F.softplus(raw[:, 1:2]) + 1e-4
        return mu, sigma

    def nll(self, raw: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        mu, sigma = self.split(raw)
        loss = 0.5 * ((target - mu) / sigma) ** 2 + torch.log(sigma) + 0.5 * torch.log(
            torch.tensor(2 * torch.pi))
        n_valid = mask.sum().clamp(min=1.0)
        return (loss * mask).sum() / n_valid

    @staticmethod
    def mixture_quantiles(mu: Tensor, sigma: Tensor, taus=(0.05, 0.5, 0.95),
                          iters=80) -> Tensor:
        """Quantiles of the uniform mixture of T gaussians, by bisection.

        mu, sigma: (batch, T) component parameters. z must stay 2-D: an
        extra singleton axis makes the CDF average that axis, and the
        bisection then solves each component's own quantile instead of the
        mixture CDF. Shapes mirror MDNStream.predict_quantiles.
        """
        lo = (mu - 12 * sigma).min(dim=-1).values - 5.0
        hi = (mu + 12 * sigma).max(dim=-1).values + 5.0
        outputs = []
        for tau in taus:
            a, b = lo.clone(), hi.clone()
            for _ in range(iters):
                mid = 0.5 * (a + b)
                z = (mid.unsqueeze(-1) - mu) / sigma
                cdf = (0.5 * (1 + torch.erf(z / SQRT2))).mean(dim=-1)
                low = cdf < tau
                a = torch.where(low, mid, a)
                b = torch.where(low, b, mid)
            outputs.append(0.5 * (a + b))
        return torch.stack(outputs, dim=-1)

"""
utils.py  -  Project-level shared helpers for LPU-Stream.

Centralises small utilities that were previously duplicated across the
experiment scripts so that every script selects the device and seeds RNGs
in the same way.
"""

import numpy as np
import torch


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU.

    Use this instead of hard-coding ``torch.device('cuda')`` so the scripts
    also run on CPU-only machines.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducibility.

    Args:
        seed: integer seed applied to torch (CPU + all CUDA devices) and NumPy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

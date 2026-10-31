"""
utils.py  -  Project-level shared helpers for LPU-Stream.

Centralises small utilities that were previously duplicated across the
experiment scripts so that every script selects the device and seeds RNGs
in the same way.
"""

import hashlib

import random

from pathlib import Path

import numpy as np
import torch


def repository_path(root: Path, value: str) -> Path:
    """Resolve a repository-relative artifact path written by any OS.

    Rejects empty segments, dot segments, drive letters, and anything that
    resolves outside ``root``; accepts the Windows separators an earlier
    generation of artifacts embedded.
    """
    parts = value.replace("\\", "/").split("/")
    if (
        not value
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    path = root.joinpath(*parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {value!r}") from exc
    return path


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU.

    Use this instead of hard-coding ``torch.device('cuda')`` so the scripts
    also run on CPU-only machines.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducibility.

    Args:
        seed: integer seed applied to torch (CPU + all CUDA devices) and NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path) -> str:
    """SHA-256 hex digest of a file, streamed so large checkpoints stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def seeded_generator(seed: int) -> torch.Generator:
    """Create a deterministic generator for shuffled DataLoaders."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

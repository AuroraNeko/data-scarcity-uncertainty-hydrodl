"""GPU-resident window gather for the Tier 3 trainings.

On this workstation, any CPU-side gap between GPU batches (loader fetch,
pipe deserialization, GIL) drains the CUDA queue, and re-submitting work
after the driver hands the GPU to competing desktop contexts costs hundreds
of milliseconds per batch. Holding each split's arrays on the GPU and
assembling windows there removes every such gap: the training loop becomes
the same back-to-back kernel stream that benchmarks at ~52 ms per batch.

Window assembly reproduces CamelsDataset.__getitem__ exactly: dynamic slice
[start, start+seq_len), target/mask at start+seq_len, and the basin's static
vector.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class GpuSplitData:
    """Flat GPU storage for one CamelsDataset split with window gather."""

    def __init__(self, dataset, device, seq_len: int):
        self.seq_len = seq_len
        self.device = device
        dyn, tgt, msk, stat = [], [], [], []
        starts, tstarts, basins = [], [], []
        offset = 0
        for bidx, bd in enumerate(dataset.basin_data):
            length = bd["dynamic"].shape[0]
            dyn.append(torch.from_numpy(np.ascontiguousarray(bd["dynamic"])))
            tgt.append(torch.from_numpy(np.ascontiguousarray(bd["target"], dtype=np.float32)))
            msk.append(torch.from_numpy(np.ascontiguousarray(bd["mask"], dtype=np.float32)))
            stat.append(torch.from_numpy(np.ascontiguousarray(bd["static"])))
            n = length - seq_len
            starts.append(torch.arange(offset, offset + n, dtype=torch.long))
            tstarts.append(torch.arange(offset + seq_len, offset + length, dtype=torch.long))
            basins.append(torch.full((n,), bidx, dtype=torch.long))
            offset += length
        self.flat_dyn = torch.cat(dyn).to(device)
        self.flat_tgt = torch.cat(tgt).to(device)
        self.flat_mask = torch.cat(msk).to(device)
        self.static = torch.stack(stat).to(device)
        self.win_start = torch.cat(starts).to(device)
        self.win_target = torch.cat(tstarts).to(device)
        self.win_basin = torch.cat(basins).to(device)
        self.arange = torch.arange(seq_len, device=device)
        self.n = int(self.win_start.shape[0])
        self.basin_ids = list(dataset.basin_ids)
        dates = np.empty(self.n, dtype="datetime64[D]")
        for sample, (bidx, start) in enumerate(dataset.indices):
            dates[sample] = dataset.basin_data[bidx]["dates"][start + seq_len]
        self.target_dates = dates
        self._generator = None

    def batches(self, batch_size: int, shuffle: bool = False, seed: int | None = None):
        if shuffle:
            if self._generator is None:
                self._generator = torch.Generator(device=self.device)
                self._generator.manual_seed(seed if seed is not None else 0)
            order = torch.randperm(self.n, device=self.device, generator=self._generator)
        else:
            order = torch.arange(self.n, device=self.device)
        for i in range(0, self.n, batch_size):
            idx = order[i:i + batch_size]
            rows = self.win_start[idx].unsqueeze(1) + self.arange.unsqueeze(0)
            dynamic = self.flat_dyn[rows]                       # (B, seq, 15)
            static = self.static[self.win_basin[idx]]           # (B, 15)
            target = self.flat_tgt[self.win_target[idx]].unsqueeze(1)
            mask = self.flat_mask[self.win_target[idx]].unsqueeze(1)
            yield dynamic, static, target, mask

    def sample_basin_array(self) -> np.ndarray:
        return np.asarray(self.basin_ids)[self.win_basin.cpu().numpy()]

    def save(self, path):
        """Persist the GPU mirror so a clean training process can load it.

        Building the 671-basin datasets inside the training process leaves
        CUDA permanently ~14x slower on this workstation; a short-lived
        builder process therefore materializes the mirror to disk, and the
        training process never touches the dataset/pandas code path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            flat_dyn=self.flat_dyn.cpu().numpy(),
            flat_tgt=self.flat_tgt.cpu().numpy(),
            flat_mask=self.flat_mask.cpu().numpy(),
            static=self.static.cpu().numpy(),
            win_start=self.win_start.cpu().numpy(),
            win_target=self.win_target.cpu().numpy(),
            win_basin=self.win_basin.cpu().numpy(),
            target_dates=self.target_dates,
        )
        (path.with_suffix(".basins.json")).write_text(
            json.dumps(self.basin_ids), encoding="utf-8")

    @classmethod
    def load(cls, path, device, seq_len: int):
        data = np.load(path, allow_pickle=False)
        mirror = cls.__new__(cls)
        mirror.seq_len = seq_len
        mirror.device = device
        mirror.flat_dyn = torch.from_numpy(data["flat_dyn"]).to(device)
        mirror.flat_tgt = torch.from_numpy(data["flat_tgt"]).to(device)
        mirror.flat_mask = torch.from_numpy(data["flat_mask"]).to(device)
        mirror.static = torch.from_numpy(data["static"]).to(device)
        mirror.win_start = torch.from_numpy(data["win_start"]).to(device)
        mirror.win_target = torch.from_numpy(data["win_target"]).to(device)
        mirror.win_basin = torch.from_numpy(data["win_basin"]).to(device)
        mirror.arange = torch.arange(seq_len, device=device)
        mirror.target_dates = data["target_dates"]
        mirror.n = int(mirror.win_start.shape[0])
        mirror.basin_ids = json.loads(
            path.with_suffix(".basins.json").read_text(encoding="utf-8"))
        mirror._generator = None
        return mirror

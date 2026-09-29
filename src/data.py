"""Data loading and normalization for the Burgers' dataset."""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class Normalizer:
    """Standardize with training-set mean/std; invertible for un-normalizing outputs."""

    def __init__(self, x, eps=1e-6):
        self.mean = x.mean()
        self.std = x.std().clamp_min(eps)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def load_split(path, grid, split, n=None):
    """Load one split ('train'/'test') at a given grid resolution.

    Returns:
      x: (N, grid, 2)  -- [field u0, x-coordinate]
      y: (N, grid, 1)  -- target field uT
    """
    data = np.load(path)
    u0 = torch.from_numpy(data[f"u0_{split}_{grid}"]).float()   # (N, grid)
    uT = torch.from_numpy(data[f"uT_{split}_{grid}"]).float()   # (N, grid)
    if n is not None:
        u0, uT = u0[:n], uT[:n]

    N = u0.shape[0]
    # The coordinate channel: same 0..1 ramp for every sample.
    coord = torch.linspace(0, 1, grid).view(1, grid).expand(N, grid)

    x = torch.stack([u0, coord], dim=-1)   # (N, grid, 2)
    y = uT.unsqueeze(-1)                    # (N, grid, 1)
    return x, y


def make_loader(x, y, batch_size=32, shuffle=True):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)
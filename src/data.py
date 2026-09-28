"""Data loading and normalization for the Burgers' and Navier-Stokes datasets."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class GaussianNormalizer:
    """Standardize with the training-set mean and std; invertible for outputs."""

    def __init__(self, x: torch.Tensor, eps: float = 1e-6):
        self.mean = x.mean()
        self.std = x.std().clamp_min(eps)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def load_burgers(path: str, grid: int | None = None, n_train: int = 1000, n_test: int = 200):
    """Load a Burgers' .npz file at a chosen resolution.

    If `grid` is given and a downsampled copy `u0_<grid>` exists it is used;
    otherwise the full-resolution arrays are returned. Adds a positional channel
    so inputs are (n, grid, 2): [field, x-coordinate].
    """
    data = np.load(path)
    if grid is not None and f"u0_{grid}" in data:
        u0, uT = data[f"u0_{grid}"], data[f"uT_{grid}"]
    else:
        u0, uT = data["u0"], data["uT"]
    g = u0.shape[-1]

    u0 = torch.from_numpy(u0).float()
    uT = torch.from_numpy(uT).float()

    xs = torch.linspace(0, 1, g).view(1, g, 1).expand(u0.shape[0], g, 1)
    x = torch.stack([u0, xs.squeeze(-1)], dim=-1)  # (n, g, 2)
    y = uT.unsqueeze(-1)                            # (n, g, 1)

    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:n_train + n_test], y[n_train:n_train + n_test]
    return (x_tr, y_tr), (x_te, y_te)


def make_loaders(x_tr, y_tr, x_te, y_te, batch_size: int = 32):
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    return tr, te

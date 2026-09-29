"""Data loading for the 2D Navier-Stokes single-step task.

Forms (omega(t), omega(t+1)) pairs from vorticity trajectories, attaches two
coordinate channels to the input, and normalizes with training statistics.
The model learns omega(t) -> omega(t+1), applied autoregressively for a rollout.
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class Normalizer:
    """Standardize with training mean/std; invertible for decoding predictions."""

    def __init__(self, x, eps=1e-6):
        self.mean = x.mean()
        self.std = x.std().clamp_min(eps)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def add_coords(field):
    """(n, N, N, 1) -> (n, N, N, 3): append x and y coordinate channels."""
    n, N = field.shape[0], field.shape[1]
    gx = torch.linspace(0, 1, N).view(1, N, 1, 1).expand(n, N, N, 1)
    gy = torch.linspace(0, 1, N).view(1, 1, N, 1).expand(n, N, N, 1)
    return torch.cat([field, gx, gy], dim=-1)


def single_step_pairs(vort):
    """(n_traj, T, N, N) -> input (M, N, N, 1), target (M, N, N, 1)."""
    x = vort[:, :-1]                      # omega(t)
    y = vort[:, 1:]                       # omega(t+1)
    N = vort.shape[-1]
    return x.reshape(-1, N, N, 1), y.reshape(-1, N, N, 1)


def load_ns(path, n_train_traj=800, n_test_traj=200):
    """Load NS vorticity .npz -> (x_tr, y_tr), (x_te, y_te), (x_norm, y_norm).

    Split by TRAJECTORY (not by pair) to avoid leakage. x has 3 channels
    (normalized vorticity + x,y coords); y_tr is normalized, y_te left raw so
    evaluation decodes the prediction back to physical units.
    """
    vort = torch.from_numpy(np.load(path)["vorticity"]).float()   # (n_traj, T, N, N)
    train = vort[:n_train_traj]
    test = vort[n_train_traj:n_train_traj + n_test_traj]

    x_tr, y_tr = single_step_pairs(train)
    x_te, y_te = single_step_pairs(test)

    x_norm = Normalizer(x_tr)
    y_norm = Normalizer(y_tr)
    x_tr = add_coords(x_norm.encode(x_tr))
    x_te = add_coords(x_norm.encode(x_te))
    y_tr = y_norm.encode(y_tr)                                     # y_te stays raw
    return (x_tr, y_tr), (x_te, y_te), (x_norm, y_norm)


def make_loader(x, y, batch_size=32, shuffle=True):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

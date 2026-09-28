"""Fourier Neural Operator (FNO), 1D and 2D, implemented from scratch.

The core building block is a spectral convolution: transform to Fourier space,
keep the lowest `modes` frequencies, multiply them by learned complex weights,
and transform back. Because the operation is defined on frequencies rather than
grid points, a trained FNO can be evaluated on a different resolution than it
was trained on.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial
Differential Equations", ICLR 2021 (arXiv:2010.08895).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D spectral convolution keeping the lowest `modes` Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Complex weights for the retained modes.
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, grid)
        batch, _, grid = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=grid, dim=-1)


class FNO1d(nn.Module):
    """1D FNO: lift -> [spectral conv + pointwise skip] x depth -> project."""

    def __init__(self, modes: int = 16, width: int = 64, depth: int = 4,
                 in_channels: int = 2, out_channels: int = 1):
        super().__init__()
        self.width = width
        # Input is the field plus a positional (grid) channel by default.
        self.fc_in = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.skips = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc_out = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, grid, in_channels)
        x = self.fc_in(x).permute(0, 2, 1)  # -> (batch, width, grid)
        for spec, skip in zip(self.spectral, self.skips):
            x = F.gelu(spec(x) + skip(x))
        x = x.permute(0, 2, 1)  # -> (batch, grid, width)
        return self.fc_out(x)


class SpectralConv2d(nn.Module):
    """2D spectral convolution keeping the lowest modes in each dimension."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # Two weight tensors: one for the low-low corner, one for the high-low corner.
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, h, w)
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(batch, self.w1.size(1), h, w // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))


class FNO2d(nn.Module):
    """2D FNO for problems like Navier-Stokes vorticity prediction."""

    def __init__(self, modes1: int = 12, modes2: int = 12, width: int = 32, depth: int = 4,
                 in_channels: int = 12, out_channels: int = 1):
        super().__init__()
        self.fc_in = nn.Linear(in_channels + 2, width)  # + 2 positional channels
        self.spectral = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.skips = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc_out = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))

    def _grid(self, shape, device):
        batch, h, w, _ = shape
        gx = torch.linspace(0, 1, h, device=device)
        gy = torch.linspace(0, 1, w, device=device)
        gx = gx.view(1, h, 1, 1).expand(batch, h, w, 1)
        gy = gy.view(1, 1, w, 1).expand(batch, h, w, 1)
        return torch.cat((gx, gy), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, h, w, in_channels)
        grid = self._grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc_in(x).permute(0, 3, 1, 2)  # -> (batch, width, h, w)
        for spec, skip in zip(self.spectral, self.skips):
            x = F.gelu(spec(x) + skip(x))
        x = x.permute(0, 2, 3, 1)  # -> (batch, h, w, width)
        return self.fc_out(x)

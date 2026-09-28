"""U-Net baselines (1D and 2D) with parameter counts comparable to the FNO.

A conventional convolutional baseline. Unlike the FNO it is tied to the grid
resolution it was trained on, which is exactly the contrast the resolution-
transfer experiment is meant to expose.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block1d(cin, cout):
    return nn.Sequential(
        nn.Conv1d(cin, cout, 3, padding=1), nn.GELU(),
        nn.Conv1d(cout, cout, 3, padding=1), nn.GELU(),
    )


class UNet1d(nn.Module):
    """Small 1D U-Net mapping (batch, grid, in_channels) -> (batch, grid, out_channels)."""

    def __init__(self, in_channels: int = 2, out_channels: int = 1, width: int = 32):
        super().__init__()
        self.enc1 = _block1d(in_channels, width)
        self.enc2 = _block1d(width, width * 2)
        self.pool = nn.MaxPool1d(2)
        self.bott = _block1d(width * 2, width * 4)
        self.up2 = nn.ConvTranspose1d(width * 4, width * 2, 2, stride=2)
        self.dec2 = _block1d(width * 4, width * 2)
        self.up1 = nn.ConvTranspose1d(width * 2, width, 2, stride=2)
        self.dec1 = _block1d(width * 2, width)
        self.head = nn.Conv1d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # -> (batch, channels, grid)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1).permute(0, 2, 1)


def _block2d(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GELU(),
    )


class UNet2d(nn.Module):
    """Small 2D U-Net mapping (batch, h, w, in_channels) -> (batch, h, w, out_channels)."""

    def __init__(self, in_channels: int = 12, out_channels: int = 1, width: int = 32):
        super().__init__()
        self.enc1 = _block2d(in_channels, width)
        self.enc2 = _block2d(width, width * 2)
        self.pool = nn.MaxPool2d(2)
        self.bott = _block2d(width * 2, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = _block2d(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = _block2d(width * 2, width)
        self.head = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # -> (batch, channels, h, w)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1).permute(0, 2, 3, 1)

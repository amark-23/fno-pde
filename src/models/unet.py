"""1D U-Net baseline for the Burgers' operator-learning task.

Configurable depth (`n_levels`), with GroupNorm + residual blocks -- a modern,
good-faith baseline (bare conv stacks underfit; normalization and residuals are
standard and greatly ease optimization). This makes the FNO-vs-U-Net comparison
fair rather than a strawman.

Requires the grid to be divisible by 2**n_levels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(c):
    """Largest group count (<=8) that divides c, for GroupNorm."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Two conv layers with GroupNorm + GELU, plus a residual skip."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class UNet1d(nn.Module):
    """Configurable-depth 1D U-Net with GroupNorm + residual blocks."""

    def __init__(self, in_channels=2, out_channels=1, width=24, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        chs = [width * (2 ** i) for i in range(n_levels)]
        bott_ch = width * (2 ** n_levels)

        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chs:
            self.enc.append(ResBlock(cin, c)); cin = c
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ResBlock(chs[-1], bott_ch)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = bott_ch
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(cin, c, 2, stride=2))
            self.dec.append(ResBlock(2 * c, c))
            cin = c

        self.head = nn.Conv1d(chs[0], out_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)                      # (B, channels, grid)

        skips = []
        h = x
        for enc in self.enc:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)
        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))

        return self.head(h).permute(0, 2, 1)        # (B, grid, out_ch)

class ResBlock2d(nn.Module):
    """Two Conv2d + GroupNorm + GELU layers with a residual skip."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class UNet2d(nn.Module):
    """Configurable-depth 2D U-Net with GroupNorm + residual blocks.
    Requires the grid divisible by 2**n_levels."""

    def __init__(self, in_channels=3, out_channels=1, width=16, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        chs = [width * (2 ** i) for i in range(n_levels)]
        bott_ch = width * (2 ** n_levels)

        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chs:
            self.enc.append(ResBlock2d(cin, c)); cin = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResBlock2d(chs[-1], bott_ch)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = bott_ch
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(ResBlock2d(2 * c, c))
            cin = c

        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)                     # (B, C, H, W)
        skips = []
        h = x
        for enc in self.enc:
            h = enc(h); skips.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.head(h).permute(0, 2, 3, 1)       # (B, H, W, out)

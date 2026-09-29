"""
Fourier Neural Operator (FNO), 1D.

Purpose
    A neural network that learns the solution operator of Burgers': it maps an
    initial field u(x, 0) directly to u(x, 1) in one pass, trained by
    backpropagation against the solver's output.

Core idea
    The solver applies a fixed operator (multiply by ik) in Fourier space. The
    FNO uses the same FFT transform but multiplies by learned weights. That
    trainable Fourier-space multiply is the spectral convolution. Because it
    acts on frequencies rather than grid points, one trained model runs at any
    resolution.

Architecture
    Input u(x, 0), shape (batch, grid, in_channels).
    1. Lift: a Linear layer raises in_channels to width.
    2. Fourier layers, repeated for depth: each computes
       spectral_conv(x) + pointwise_linear(x), then a GELU. spectral_conv keeps
       only the low modes; pointwise_linear is a 1x1 conv carrying all
       frequencies.
    3. Project: two Linear layers reduce width to out_channels.
    Output u(x, 1) prediction, shape (batch, grid, out_channels).

Classes
    SpectralConv1d: the learned Fourier-space multiply.
    FNO1d: lift, stacked Fourier layers, project.

Input convention
    The field is supplied with its x-coordinate as a second channel, so
    in_channels = 2. Providing the grid coordinate is a standard aid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Learned Fourier-space multiply, keeping the lowest `modes` frequencies.

    FFT the input, keep the lowest `modes` Fourier coefficients, multiply them
    by learned complex weights, zero the rest, inverse-FFT back.

    
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes                       # how many low modes to keep

        # Learned complex weights R, one (in,out) matrix per kept mode.
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (batch, in_channels, grid)
        batch, _, grid = x.shape

        # 1. FFT along the spatial axis (real FFT -> nonneg frequencies only).
        x_ft = torch.fft.rfft(x, dim=-1)          # (batch, in_channels, grid//2 + 1)

        # 2. Prepare an all-zeros output spectrum, then fill only the low modes.
        out_ft = torch.zeros(batch, self.out_channels, x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        m = min(self.modes, x_ft.size(-1))        # guard: don't exceed available modes

        # 3. Multiply the kept low modes by the learned weights R.
        #    einsum over channels: (batch,in,m) x (in,out,m) -> (batch,out,m)
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom",
                                        x_ft[:, :, :m], self.weight[:, :, :m])

        # 4. Inverse FFT back to a field at the original grid size.
        return torch.fft.irfft(out_ft, n=grid, dim=-1)    


class FNO1d(nn.Module):
    """Full 1D FNO: lift -> [spectral conv + linear skip + GELU] x depth -> project."""

    def __init__(self, modes=16, width=64, depth=4, in_channels=2, out_channels=1):
        super().__init__()
        self.width = width

        # Lift: raise input channels to the hidden width.
        self.fc_in = nn.Linear(in_channels, width)

        # depth Fourier layers, each: spectral conv (global) + 1x1 conv (local).
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.local    = nn.ModuleList([nn.Conv1d(width, width, 1)          for _ in range(depth)])

        # Project: hidden width -> out_channels.
        self.fc_out = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels)
        )

    def forward(self, x):
        # x: (batch, grid, in_channels)
        x = self.fc_in(x)              # -> (batch, grid, width)
        x = x.permute(0, 2, 1)         # -> (batch, width, grid)  [channels-first for the convs]

        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))   # global path + local path, then nonlinearity

        x = x.permute(0, 2, 1)         # -> (batch, grid, width)  [back to channels-last]
        return self.fc_out(x)          # -> (batch, grid, out_channels)



class SpectralConv2d(nn.Module):
    """2D learned Fourier-space multiply, keeping the lowest (modes1, modes2)."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # Two weight blocks: low-positive-kx and low-negative-kx corners.
        self.weight1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        # x: (batch, in_channels, H, W)
        batch, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)                       # (batch, in, H, W//2 + 1)

        out_ft = torch.zeros(batch, self.weight1.shape[1], H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        # top-left corner: low +kx, low ky
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight1)
        # bottom-left corner: low -kx, low ky
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weight2)

        return torch.fft.irfft2(out_ft, s=(H, W))       # back to (batch, out, H, W)


class FNO2d(nn.Module):
    """2D FNO: lift -> [spectral conv + 1x1 conv + GELU] x depth -> project."""

    def __init__(self, modes1=12, modes2=12, width=32, depth=4,
                 in_channels=3, out_channels=1):
        super().__init__()
        self.fc_in = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)])
        self.local = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc_out = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))

    def forward(self, x):
        # x: (batch, H, W, in_channels)
        x = self.fc_in(x).permute(0, 3, 1, 2)          # -> (batch, width, H, W)
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))               # global + local, then nonlinearity
        x = x.permute(0, 2, 3, 1)                       # -> (batch, H, W, width)
        return self.fc_out(x)                           # -> (batch, H, W, out_channels)

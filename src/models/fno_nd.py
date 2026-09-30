"""
n-dimensional Fourier spectral convolution.

A single spectral convolution that works in any number of spatial dimensions,
with the 1D and 2D versions in fno.py recovered as the cases d = 1 and d = 2.
The layer keeps the lowest modes on each axis, multiplies them by learned complex
weights, and transforms back, exactly as the fixed-dimension versions do.

The only part that depends on the dimension is the mode bookkeeping. The real FFT
(rfftn) stores only the non-negative frequencies of its last axis, so that axis
contributes a single low block. Each of the other d - 1 axes keeps both signs, so
its low modes appear at both ends of the axis. Selecting the low block at each end
of those axes gives 2**(d - 1) corner blocks, and each corner carries its own
weight tensor.
"""
import itertools

import torch
import torch.nn as nn


class SpectralConvNd(nn.Module):
    """Spectral convolution in d spatial dimensions.

    modes is a sequence of length d giving the number of low modes kept on each
    spatial axis. The weights have shape (2**(d-1), in_channels, out_channels,
    *modes): one (in, out, *modes) block per corner.
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = tuple(modes)
        self.d = len(self.modes)
        self.n_corners = 2 ** (self.d - 1)

        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.rand(self.n_corners, in_channels, out_channels,
                               *self.modes, dtype=torch.cfloat))

    def _corner_slices(self):
        """Yield (corner_index, spatial_slice_tuple) for the retained blocks.

        For the first d - 1 axes each corner selects the low-positive block
        slice(0, m) or the low-negative block slice(-m, None); the last axis,
        halved by the real FFT, always takes slice(0, m_last).
        """
        head = self.modes[:-1]
        last = self.modes[-1]
        for idx, ends in enumerate(itertools.product((0, 1), repeat=self.d - 1)):
            sl = [slice(0, head[ax]) if end == 0 else slice(-head[ax], None)
                  for ax, end in enumerate(ends)]
            sl.append(slice(0, last))
            yield idx, tuple(sl)

    def _einsum(self):
        letters = "xyzwvu"[:self.d]
        return f"bi{letters},io{letters}->bo{letters}"

    def forward(self, x):
        # x: (batch, in_channels, *spatial) with d spatial axes
        spatial = x.shape[2:]
        dims = tuple(range(2, 2 + self.d))
        x_ft = torch.fft.rfftn(x, dim=dims)

        out_ft = torch.zeros(x.shape[0], self.out_channels, *x_ft.shape[2:],
                             dtype=torch.cfloat, device=x.device)
        eq = self._einsum()
        for idx, sl in self._corner_slices():
            block = (slice(None), slice(None)) + sl
            out_ft[block] = torch.einsum(eq, x_ft[block], self.weight[idx])

        return torch.fft.irfftn(out_ft, s=spatial, dim=dims)

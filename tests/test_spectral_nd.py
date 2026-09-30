"""Equivalence and smoke tests for SpectralConvNd.

Checks that the n-dimensional spectral convolution reproduces SpectralConv1d and
SpectralConv2d exactly when given the same weights, and that it runs a forward and
backward pass in 3D.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.fno import SpectralConv1d, SpectralConv2d
from src.models.fno_nd import SpectralConvNd


def test_matches_1d():
    torch.manual_seed(0)
    ref = SpectralConv1d(3, 5, modes=8)
    nd = SpectralConvNd(3, 5, modes=(8,))
    with torch.no_grad():
        nd.weight.copy_(ref.weight.unsqueeze(0))          # (1, in, out, modes)
    x = torch.randn(2, 3, 64)
    a, b = ref(x), nd(x)
    err = ((a - b).norm() / a.norm()).item()
    assert torch.allclose(a, b, atol=1e-5), f"1D mismatch, rel {err:.2e}"
    return err


def test_matches_2d():
    torch.manual_seed(0)
    ref = SpectralConv2d(3, 5, modes1=6, modes2=8)
    nd = SpectralConvNd(3, 5, modes=(6, 8))
    with torch.no_grad():
        nd.weight[0].copy_(ref.weight1)                   # low +kx corner
        nd.weight[1].copy_(ref.weight2)                   # low -kx corner
    x = torch.randn(2, 3, 32, 40)
    a, b = ref(x), nd(x)
    err = ((a - b).norm() / a.norm()).item()
    assert torch.allclose(a, b, atol=1e-5), f"2D mismatch, rel {err:.2e}"
    return err


def test_runs_3d():
    nd = SpectralConvNd(2, 4, modes=(4, 4, 4))
    x = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
    y = nd(x)
    assert y.shape == (2, 4, 8, 8, 8), f"3D shape {tuple(y.shape)}"
    y.sum().backward()
    assert nd.weight.grad is not None and x.grad is not None
    return tuple(y.shape)


if __name__ == "__main__":
    e1 = test_matches_1d(); print(f"1D equivalence: rel-L2 {e1:.2e}  OK")
    e2 = test_matches_2d(); print(f"2D equivalence: rel-L2 {e2:.2e}  OK")
    s3 = test_runs_3d();    print(f"3D forward+backward: output {s3}  OK")
    print("all passed")

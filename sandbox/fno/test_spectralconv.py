"""Sandbox test: SpectralConv1d. Run in your venv:  py sandbox/fno/test_spectralconv.py"""
import os, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("fno", os.path.join(REPO, "src", "models", "fno.py"))
fno = importlib.util.module_from_spec(spec); spec.loader.exec_module(fno)

layer = fno.SpectralConv1d(in_channels=3, out_channels=5, modes=16)
n_params = sum(p.numel() for p in layer.parameters())
print("params:", n_params)
print("weight dtype:", layer.weight.dtype, "| shape:", tuple(layer.weight.shape))

# The key test: SAME layer, two resolutions.
for grid in (64, 256):
    x = torch.randn(4, 3, grid)           # (batch, in_channels, grid)
    y = layer(x)
    print(f"grid {grid:4d}:  in {tuple(x.shape)}  ->  out {tuple(y.shape)}   real={y.is_floating_point()}")

# Does backprop reach the learned weights?
x = torch.randn(2, 3, 128)
layer(x).sum().backward()
print("gradient reaches weights:", layer.weight.grad is not None)
print("ALL OK")

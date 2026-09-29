"""Sandbox test: full FNO1d. Run in your venv:  py sandbox/fno/test_fno.py"""
import os, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("fno", os.path.join(REPO, "src", "models", "fno.py"))
fno = importlib.util.module_from_spec(spec); spec.loader.exec_module(fno)

model = fno.FNO1d(modes=16, width=64, depth=4, in_channels=2, out_channels=1)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"FNO1d trainable params: {n_params:,}")

# Input convention: (batch, grid, in_channels) = [field, x-coordinate].
def make_input(batch, grid):
    field = torch.randn(batch, grid, 1)
    xcoord = torch.linspace(0, 1, grid).view(1, grid, 1).expand(batch, grid, 1)
    return torch.cat([field, xcoord], dim=-1)   # (batch, grid, 2)

# Runs end-to-end at multiple resolutions with the SAME weights.
for grid in (64, 128, 256):
    x = make_input(4, grid)
    y = model(x)
    print(f"grid {grid:4d}:  in {tuple(x.shape)}  ->  out {tuple(y.shape)}")

# Gradients flow through the whole model?
x = make_input(2, 128)
loss = model(x).pow(2).mean()
loss.backward()
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
n_total = sum(1 for _ in model.parameters())
print(f"params receiving gradient: {n_with_grad}/{n_total}")
print("ALL OK")

"""Sandbox test: FNO2d. Run in venv:  py sandbox/ns/test_fno2d.py"""
import os, importlib.util
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("fno", os.path.join(REPO, "src", "models", "fno.py"))
fno = importlib.util.module_from_spec(spec); spec.loader.exec_module(fno)

# SpectralConv2d alone: same weights at two resolutions.
sc = fno.SpectralConv2d(4, 5, modes1=8, modes2=8)
for N in (32, 64):
    x = torch.randn(2, 4, N, N)
    print(f"SpectralConv2d @ {N}: in {tuple(x.shape)} -> out {tuple(sc(x).shape)}")

# Full FNO2d.
model = fno.FNO2d(modes1=12, modes2=12, width=32, depth=4, in_channels=3, out_channels=1)
n = sum(p.numel() for p in model.parameters())
print(f"FNO2d params: {n:,}")

def make_input(batch, N):
    field = torch.randn(batch, N, N, 1)
    gx = torch.linspace(0, 1, N).view(1, N, 1, 1).expand(batch, N, N, 1)
    gy = torch.linspace(0, 1, N).view(1, 1, N, 1).expand(batch, N, N, 1)
    return torch.cat([field, gx, gy], dim=-1)          # (batch, N, N, 3)

for N in (32, 64, 128):
    x = make_input(2, N)
    y = model(x)
    print(f"FNO2d @ {N}: in {tuple(x.shape)} -> out {tuple(y.shape)}")

# gradients flow?
make_input(2, 64)
model(make_input(2, 64)).pow(2).mean().backward()
print("gradients ok:", all(p.grad is not None for p in model.parameters()))
print("ALL OK")

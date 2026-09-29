"""Sandbox test: 3-level UNet1d. Run in venv:  py sandbox/fno/test_unet.py"""
import os, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("unet", os.path.join(REPO, "src", "models", "unet.py"))
unet = importlib.util.module_from_spec(spec); spec.loader.exec_module(unet)

print("param counts vs FNO target (287,425):")
for w in (16, 20, 24, 28, 32):
    m = unet.UNet1d(in_channels=2, out_channels=1, width=w)
    n = sum(p.numel() for p in m.parameters())
    print(f"  width={w:3d}:  {n:,} params")

# Shape + receptive-field sanity at grid 256 (divisible by 8).
model = unet.UNet1d(width=24)
x = torch.randn(4, 256, 2)
y = model(x)
print("grid 256:  in", tuple(x.shape), "-> out", tuple(y.shape))
model(x).pow(2).mean().backward()
print("gradients ok:", all(p.grad is not None for p in model.parameters()))
print("ALL OK")

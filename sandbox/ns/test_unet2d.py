"""Sandbox test: UNet2d. Run in venv:  py sandbox/ns/test_unet2d.py"""
import os, importlib.util
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location("unet", os.path.join(REPO, "src", "models", "unet.py"))
unet = importlib.util.module_from_spec(spec); spec.loader.exec_module(unet)

print("param counts vs FNO2d target (1,188,353):")
for w in (12, 16, 20, 24):
    m = unet.UNet2d(in_channels=3, out_channels=1, width=w, n_levels=3)
    print(f"  width={w:3d}: {sum(p.numel() for p in m.parameters()):,}")

model = unet.UNet2d(in_channels=3, out_channels=1, width=16, n_levels=3)
x = torch.randn(2, 64, 64, 3)          # (B, H, W, in)
y = model(x)
print("UNet2d @ 64: in", tuple(x.shape), "-> out", tuple(y.shape))
model(x).pow(2).mean().backward()
print("gradients ok:", all(p.grad is not None for p in model.parameters()))
print("ALL OK")

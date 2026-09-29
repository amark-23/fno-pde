"""Sandbox test: src/data.py loaders. Run in venv:  py sandbox/fno/test_data.py"""
import os, importlib.util
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("data", os.path.join(REPO, "src", "data.py"))
data = importlib.util.module_from_spec(spec); spec.loader.exec_module(data)

npz = os.path.join(REPO, "data", "burgers.npz")

# Load the training split at grid 256.
x, y = data.load_split(npz, grid=256, split="train")
print("train x:", tuple(x.shape), "  y:", tuple(y.shape))
print("  channel 0 (field) range:", round(x[..., 0].min().item(), 3), "to", round(x[..., 0].max().item(), 3))
print("  channel 1 (coord) range:", round(x[..., 1].min().item(), 3), "to", round(x[..., 1].max().item(), 3))

# Normalizer round-trips.
norm = data.Normalizer(x[..., 0])
enc = norm.encode(x[..., 0])
print("normalized field mean/std:", round(enc.mean().item(), 4), round(enc.std().item(), 4))
print("decode round-trip ok:", torch.allclose(norm.decode(enc), x[..., 0], atol=1e-4))

# A DataLoader batch.
loader = data.make_loader(x, y, batch_size=32)
xb, yb = next(iter(loader))
print("one batch:", tuple(xb.shape), tuple(yb.shape))

# Multi-resolution load works too.
for g in (64, 128, 256):
    xg, yg = data.load_split(npz, grid=g, split="test")
    print(f"test @ {g}: x {tuple(xg.shape)}  y {tuple(yg.shape)}")
print("ALL OK")

"""Overfit-a-tiny-batch diagnostic. Run in venv:  py sandbox/fno/diagnose_overfit.py

Trains each model on only N_SMALL samples for many steps. A healthy model with
no bugs should drive train rel-L2 toward ~0 (pure memorization). If it can't,
the problem is structural (bug / representational wall), not data or generalization.
"""
import os, sys
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from src.data import load_split, Normalizer
from src.metrics import relative_l2
from src.models.fno import FNO1d
from src.models.unet import UNet1d

N_SMALL = 16
STEPS = 800
device = torch.device("cpu")

x, y = load_split(os.path.join(REPO, "data", "burgers.npz"), 256, "train", n=N_SMALL)
xn, yn = Normalizer(x[..., 0]), Normalizer(y)
x = x.clone(); x[..., 0] = xn.encode(x[..., 0]); y = yn.encode(y)
x, y = x.to(device), y.to(device)

# Reference "trivial" errors on this tiny set, for context.
print("=== trivial baselines (normalized target) ===")
print("  predict-zero rel-L2:", round(relative_l2(torch.zeros_like(y), y).item(), 4))
print("  predict-input(u0) rel-L2:", round(relative_l2(x[..., :1], y).item(), 4))
print()

def overfit(name, model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(STEPS):
        opt.zero_grad()
        loss = relative_l2(model(x), y)
        loss.backward(); opt.step()
        if (step + 1) % (STEPS // 8) == 0:
            print(f"  {name} step {step+1:4d}: train rel-L2 {loss.item():.4f}")
    return loss.item()

print("=== FNO (control — should reach ~0) ===")
fno_final = overfit("FNO ", FNO1d(modes=16, width=64, depth=4).to(device))
print()
print("=== U-Net (the suspect) ===")
unet_final = overfit("UNet", UNet1d(width=24).to(device))
print()
print(f"FINAL  FNO: {fno_final:.4f}   U-Net: {unet_final:.4f}")
print("Interpretation: if U-Net can't reach near 0 on 16 samples, it's structural.")

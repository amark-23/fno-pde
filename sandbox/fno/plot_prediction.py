"""Sandbox: train the FNO briefly, then plot prediction vs solver truth.
Run in venv:  py sandbox/fno/plot_prediction.py"""
import os, sys, importlib
import torch
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from src.train import train_one
from src.data import load_split

cfg = {"data": os.path.join(REPO, "data", "burgers.npz"),
       "grid": 256, "n_train": 1000, "n_test": 200,
       "modes": 16, "width": 64, "depth": 4,
       "epochs": 50, "batch_size": 32, "lr": 1e-3, "seed": 0}

device = torch.device("cpu")
print("training (50 epochs for a quick look)...")
model, row = train_one(cfg, device)
print("test rel-L2:", row["test_rel_l2"])

# NOTE: train_one normalizes internally; for a fair plot we re-encode inputs the
# same way. Simplest: reuse its normalizers by recomputing from train stats.
from src.data import Normalizer
x_tr, y_tr = load_split(cfg["data"], 256, "train", n=1000)
x_te, y_te = load_split(cfg["data"], 256, "test",  n=200)
xn = Normalizer(x_tr[..., 0]); yn = Normalizer(y_tr)
x_te = x_te.clone(); x_te[..., 0] = xn.encode(x_te[..., 0])

model.eval()
with torch.no_grad():
    pred = yn.decode(model(x_te).cpu())

x = np.linspace(0, 1, 256, endpoint=False)
fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
for i in range(4):
    ax[i].plot(x, y_te[i, :, 0], lw=3, alpha=0.45, label="solver truth uT")
    ax[i].plot(x, pred[i, :, 0], "--", lw=1.6, label="FNO prediction")
    err = torch.linalg.norm(pred[i]-y_te[i]) / torch.linalg.norm(y_te[i])
    ax[i].set_title(f"test sample {i+1}   rel-L2={err:.4f}")
    ax[i].set_xlabel("x")
ax[0].set_ylabel("u(x,1)"); ax[0].legend(fontsize=8)
fig.suptitle("FNO prediction vs solver ground truth (grid 256)")
plt.tight_layout()
out = os.path.join(HERE, "prediction_vs_truth.png")
plt.savefig(out, dpi=120)
print("saved", out)

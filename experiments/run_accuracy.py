"""Experiment 1 — Accuracy: FNO vs U-Net at grid 256.
Run:  py experiments/run_accuracy.py"""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _common import (train_model, eval_at, n_params, load_split,
                     FIGDIR, NPZ, device)

EPOCHS = 250
base = dict(grid=256, n_train=1000, n_test=200, epochs=EPOCHS,
            batch_size=32, lr=1e-3, weight_decay=1e-4, lr_step=100, lr_gamma=0.5, seed=0)
configs = {
    "FNO":   {**base, "model": "fno",  "modes": 16, "width": 64, "depth": 4},
    "U-Net": {**base, "model": "unet", "width": 24, "n_levels": 3},
}

print(f"Device: {device}  |  {EPOCHS} epochs each")
res, models = {}, {}
for name, cfg in configs.items():
    m, xn, yn = train_model(cfg, 256, epochs=EPOCHS)
    err = eval_at(m, xn, yn, 256)
    res[name] = (err, n_params(m)); models[name] = (m, xn, yn)
    print(f"{name:6s}  test rel-L2 = {err:.4f}   params = {n_params(m):,}")

x_te, y_te = load_split(NPZ, 256, "test")
xg = np.linspace(0, 1, 256, endpoint=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

names = list(res); errs = [res[n][0] for n in names]
ax[0].bar(names, errs, color=["#3b6", "#c73"])
for i, e in enumerate(errs):
    ax[0].text(i, e, f"{e:.4f}", ha="center", va="bottom")
ax[0].set_ylabel("test rel-L2"); ax[0].set_title("Accuracy @ grid 256 (lower = better)")

i = 0
ax[1].plot(xg, y_te[i, :, 0], lw=3, alpha=0.4, label="truth")
for name in names:
    m, xn, yn = models[name]
    xin = x_te.clone(); xin[..., 0] = xn.encode(xin[..., 0])
    with torch.no_grad():
        p = yn.decode(m(xin.to(device)).cpu())[i, :, 0]
    ax[1].plot(xg, p, "--", lw=1.5, label=name)
ax[1].set_title("Prediction vs truth (test sample 0)"); ax[1].set_xlabel("x"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{FIGDIR}/exp1_accuracy.png", dpi=120)
print(f"saved {FIGDIR}/exp1_accuracy.png")

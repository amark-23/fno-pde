"""Experiment 2: Resolution transfer: train at grid 64, evaluate at 64/128/256.
The FNO should hold; the U-Net (grid-tied kernels) should degrade.
Run:  py experiments/run_resolution_transfer.py"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _common import train_model, eval_at, FIGDIR, device

EPOCHS = 250
TRAIN_GRID = 64
EVAL_GRIDS = [64, 128, 256]
base = dict(n_train=1000, batch_size=32, lr=1e-3, weight_decay=1e-4,
            lr_step=100, lr_gamma=0.5, seed=0)
configs = {
    "FNO":   {**base, "model": "fno",  "modes": 16, "width": 64, "depth": 4},
    "U-Net": {**base, "model": "unet", "width": 24, "n_levels": 3},
}

print(f"Device: {device}  |  train @ {TRAIN_GRID}, eval @ {EVAL_GRIDS}")
curves = {}
for name, cfg in configs.items():
    m, xn, yn = train_model(cfg, TRAIN_GRID, epochs=EPOCHS)
    curves[name] = [eval_at(m, xn, yn, g) for g in EVAL_GRIDS]
    print(f"{name:6s}  " + "  ".join(f"@{g}:{e:.4f}" for g, e in zip(EVAL_GRIDS, curves[name])))

plt.figure(figsize=(7, 4.6))
for name, ys in curves.items():
    plt.plot(EVAL_GRIDS, ys, "-o", lw=2, label=name)
plt.axvline(TRAIN_GRID, color="k", ls="--", lw=1, label=f"trained @ {TRAIN_GRID}")
plt.xlabel("evaluation grid resolution"); plt.ylabel("test rel-L2")
plt.title("Resolution transfer (trained @ 64)"); plt.legend(); plt.yscale("log")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/exp2_resolution_transfer.png", dpi=120)
print(f"saved {FIGDIR}/exp2_resolution_transfer.png")

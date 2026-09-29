"""Experiment 3 — Data efficiency: FNO test error vs training-set size.
Run:  py experiments/run_data_efficiency.py"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from _common import train_model, eval_at, FIGDIR, device

EPOCHS = 250
SIZES = [100, 250, 500, 1000]
base = dict(model="fno", modes=16, width=64, depth=4, grid=256,
            batch_size=32, lr=1e-3, weight_decay=1e-4, lr_step=100, lr_gamma=0.5, seed=0)

print(f"Device: {device}  |  FNO, sizes {SIZES}")
errs = []
for n in SIZES:
    cfg = {**base, "n_train": n}
    m, xn, yn = train_model(cfg, 256, epochs=EPOCHS)
    e = eval_at(m, xn, yn, 256); errs.append(e)
    print(f"  n_train={n:5d}  test rel-L2 = {e:.4f}")

plt.figure(figsize=(7, 4.6))
plt.plot(SIZES, errs, "-o", lw=2)
plt.xlabel("training-set size"); plt.ylabel("test rel-L2")
plt.title("FNO data efficiency (grid 256)"); plt.xscale("log"); plt.yscale("log")
plt.tight_layout(); plt.savefig(f"{FIGDIR}/exp3_data_efficiency.png", dpi=120)
print(f"saved {FIGDIR}/exp3_data_efficiency.png")

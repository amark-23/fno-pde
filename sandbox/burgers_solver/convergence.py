"""Sandbox: time-step convergence check for solve_burgers_if. Throwaway."""
import os, importlib.util
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("burgers", os.path.join(REPO, "src", "solvers", "burgers.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

grid = 256
u0 = b.gaussian_random_field(1, grid, alpha=2.5, tau=7.0, seed=3)
u0 = u0 / np.abs(u0).max()
relL2 = lambda a, c: np.linalg.norm(a - c) / np.linalg.norm(c)

# Solve at a sequence of halving time steps.
dts = [2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4]
sols = {dt: b.solve_burgers_if(u0, nu=0.01, t_final=1.0, dt=dt)[0] for dt in dts}
ref = sols[dts[-1]]   # finest dt = our reference "truth"

print(f"{'dt':>10} {'rel-L2 vs finest':>18} {'rel-L2 vs (this vs dt/2)':>26}")
errs = []
for i, dt in enumerate(dts[:-1]):
    e_ref  = relL2(sols[dt], ref)
    e_half = relL2(sols[dt], sols[dts[i + 1]])   # change when you halve dt
    errs.append(e_ref)
    print(f"{dt:10.2e} {e_ref:18.2e} {e_half:26.2e}")

plt.figure(figsize=(6.5, 4.5))
plt.loglog(dts[:-1], errs, "-o", ms=6)
plt.xlabel("dt"); plt.ylabel("rel-L2 error vs finest dt")
plt.title("Convergence of solve_burgers_if\n(error shrinks as dt decreases)")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "convergence.png"), dpi=110)
print("saved convergence.png")

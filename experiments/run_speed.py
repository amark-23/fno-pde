"""Experiment 5 — Speed: FNO inference vs the numerical solver, same batch.
Run:  py experiments/run_speed.py"""
import time, numpy as np, torch
from _common import load_split, build_model, device, NPZ
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.solvers.burgers import solve_burgers_if

GRID, N = 256, 200
x, _ = load_split(NPZ, GRID, "test", n=N)
u0 = x[..., 0].numpy()                       # (N, grid) initial fields

# Numerical solver time (produces uT from u0).
t0 = time.perf_counter()
_ = solve_burgers_if(u0, nu=0.01, t_final=1.0, dt=1e-3)
solver_t = time.perf_counter() - t0

# FNO inference time (one forward pass). Untrained is fine for timing.
model = build_model({"model": "fno", "modes": 16, "width": 64, "depth": 4}).to(device).eval()
xt = x.to(device)
with torch.no_grad():
    for _ in range(3): model(xt)              # warmup
    t0 = time.perf_counter()
    for _ in range(10): model(xt)
    fno_t = (time.perf_counter() - t0) / 10

print(f"batch of {N} @ grid {GRID}")
print(f"  numerical solver : {solver_t*1e3:8.1f} ms   ({solver_t/N*1e3:.3f} ms/sample)")
print(f"  FNO inference    : {fno_t*1e3:8.1f} ms   ({fno_t/N*1e3:.3f} ms/sample)")
print(f"  speedup (solver / FNO): {solver_t/fno_t:.1f}x")

"""Verify the torch NS solver matches the NumPy one. Run in venv:
    py sandbox/ns/test_torch_port.py"""
import os, importlib.util
import numpy as np, torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

ns_np = load("ns_np", "src/solvers/navier_stokes.py")
ns_pt = load("ns_pt", "src/solvers/navier_stokes_torch.py")

# Same initial field for both, run a short trajectory.
N = 32
rng = np.random.default_rng(0)
w0 = ns_np.gaussian_random_field_2d(3, N, seed=0)        # (3, N, N) numpy

out_np = ns_np.solve_ns(w0, nu=1e-3, t_final=0.5, dt=1e-3, save_every=100)      # (n, 3, N, N)

w0_t = torch.tensor(w0, dtype=torch.float64)             # float64 for a tight comparison
out_pt = ns_pt.solve_ns(w0_t, nu=1e-3, t_final=0.5, dt=1e-3, save_every=100).numpy()

print("numpy snaps:", out_np.shape, "| torch snaps:", out_pt.shape)
rel = np.linalg.norm(out_pt - out_np) / np.linalg.norm(out_np)
print(f"relative difference (torch vs numpy): {rel:.2e}")
print("MATCH" if rel < 1e-6 else ("close" if rel < 1e-3 else "MISMATCH"))

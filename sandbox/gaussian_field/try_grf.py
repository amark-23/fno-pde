"""Sandbox test: visualize gaussian_random_field. Throwaway — delete later.

Run from anywhere, e.g. from the repo root:
    py sandbox/gaussian_field/try_grf.py
"""
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Locate src/solvers/burgers.py relative to THIS file, so cwd doesn't matter.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
burgers_path = os.path.join(REPO, "src", "solvers", "burgers.py")

spec = importlib.util.spec_from_file_location("burgers", burgers_path)
burgers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(burgers)

# Draw a few curves and look at them. Change alpha / tau to explore.
U = burgers.gaussian_random_field(n_samples=4, grid=256, alpha=2.5, tau=7.0, seed=0)
print("shape:", U.shape, "| dtype:", U.dtype)

x = np.linspace(0, 1, 256, endpoint=False)
for i in range(4):
    plt.plot(x, U[i], label=f"sample {i}")
plt.xlabel("x"); plt.ylabel("u(x, 0)")
plt.title("GRF samples — try changing alpha!")
plt.legend(); plt.tight_layout()
plt.show()

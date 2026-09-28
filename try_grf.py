"""Scratch runner to see gaussian_random_field working. Safe to delete."""
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

# Load your solver file by path (no package setup needed).
spec = importlib.util.spec_from_file_location("burgers", "src/solvers/burgers.py")
burgers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(burgers)

# Draw 4 curves and look at them.
U = burgers.gaussian_random_field(n_samples=4, grid=256, alpha=2, tau=70, seed=0)
print("shape:", U.shape, "| dtype:", U.dtype)

x = np.linspace(0, 1, 256, endpoint=False)
for i in range(4):
    plt.plot(x, U[i], label=f"sample {i}")
plt.xlabel("x"); plt.ylabel("u(x, 0)")
plt.title("GRF samples — try changing alpha!")
plt.legend(); plt.tight_layout()
plt.show()   # opens a window on your machine

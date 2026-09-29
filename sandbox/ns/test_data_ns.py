"""Sandbox test: src/data_ns.py on synthetic data. Run in venv:
    py sandbox/ns/test_data_ns.py"""
import os, importlib.util
import numpy as np, torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
synth = os.path.join(REPO, "sandbox", "ns", "_synth.npz")
np.savez(synth, vorticity=np.random.randn(10, 6, 16, 16).astype("float32"))  # (n_traj, T, N, N)

spec = importlib.util.spec_from_file_location("data_ns", os.path.join(REPO, "src", "data_ns.py"))
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)

(x_tr, y_tr), (x_te, y_te), (xn, yn) = d.load_ns(synth, n_train_traj=8, n_test_traj=2)
print("x_tr:", tuple(x_tr.shape), " y_tr:", tuple(y_tr.shape))   # (8*5,16,16,3),(40,16,16,1)
print("x_te:", tuple(x_te.shape), " y_te:", tuple(y_te.shape))   # (10,16,16,3),(10,16,16,1)
print("input channels: vort mean/std", round(x_tr[...,0].mean().item(),3), round(x_tr[...,0].std().item(),3),
      "| coord range", round(x_tr[...,1].min().item(),2), "to", round(x_tr[...,1].max().item(),2))
loader = d.make_loader(x_tr, y_tr, batch_size=8)
xb, yb = next(iter(loader))
print("batch:", tuple(xb.shape), tuple(yb.shape))
os.remove(synth)
print("ALL OK")

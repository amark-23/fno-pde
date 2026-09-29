"""Sandbox test: experiments_ns helpers. Run in venv:
    py sandbox/ns/test_experiments_ns.py"""
import os, sys, importlib.util
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from src.experiments_ns import downsample_2d, rollout
from src.data_ns import Normalizer

# downsample_2d: 64 -> 32 on a smooth field
k = torch.fft.fftfreq(64, d=1/64); kx=k.reshape(64,1); ky=k.reshape(1,64); k2=kx**2+ky**2
scale=(k2+9.0)**-1.0; scale[0,0]=0
f = torch.fft.ifft2((torch.randn(3,64,64)+1j*torch.randn(3,64,64))*scale).real
g = downsample_2d(f, 32)
print("downsample:", tuple(f.shape), "->", tuple(g.shape), "| finite:", bool(torch.isfinite(g).all()))
# energy roughly preserved for a smooth field
print("mean-abs ratio (coarse/fine):", round((g.abs().mean()/f.abs().mean()).item(), 3))

# rollout with a dummy identity-ish model
class Dummy(torch.nn.Module):
    def forward(self, x): return x[..., :1]          # returns (B,N,N,1) = normalized input field
xn = Normalizer(torch.randn(10,16,16,1)); yn = Normalizer(torch.randn(10,16,16,1))
w0 = torch.randn(4, 16, 16)
traj = rollout(Dummy(), w0, n_steps=5, x_norm=xn, y_norm=yn)
print("rollout:", tuple(traj.shape))                 # (4, 5, 16, 16)
print("ALL OK")

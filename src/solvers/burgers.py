"""1D viscous Burgers' equation solver and dataset generation.

Solves the periodic viscous Burgers' equation on x in [0, 1):

    u_t + u u_x = nu u_xx

with a pseudo-spectral method in space (FFT) and an exponential-time-
differencing RK2 (ETDRK2) scheme in time, which handles the stiff viscous
term well. Initial conditions are drawn from a Gaussian random field so the
learned operator sees a distribution of smooth inputs.

Run directly to generate a dataset:

    python -m src.solvers.burgers --n_samples 1200 --grid 1024 --out data/burgers.npz
"""
from __future__ import annotations

import argparse
import numpy as np


def gaussian_random_field(n_samples: int, grid: int, alpha: float = 2.5,
                          tau: float = 7.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample smooth periodic initial conditions from a GRF.

    The field is defined in Fourier space with a decaying spectrum
    (tau^2 + (2 pi k)^2)^(-alpha/2); larger alpha and tau give smoother fields.

    Returns an array of shape (n_samples, grid).
    """
    rng = rng or np.random.default_rng()
    k = np.fft.fftfreq(grid, d=1.0 / grid)  # integer wavenumbers
    # Spectral density; skip the zero mode to keep fields mean-centred.
    sqrt_eig = (tau ** (alpha - 1)) * (tau ** 2 + (2 * np.pi * k) ** 2) ** (-alpha / 2)
    sqrt_eig[0] = 0.0

    noise = rng.standard_normal((n_samples, grid)) + 1j * rng.standard_normal((n_samples, grid))
    coeffs = noise * sqrt_eig[None, :] * grid
    field = np.fft.ifft(coeffs, axis=-1).real
    return field


def solve_burgers(u0: np.ndarray, nu: float, t_final: float, dt: float) -> np.ndarray:
    """Integrate Burgers' from u0 to t_final with ETDRK2.

    u0 has shape (n_samples, grid); returns the solution at t_final, same shape.
    The nonlinear term u u_x is computed as 0.5 (u^2)_x (conservative form) and
    dealiased with a 2/3-rule mask.
    """
    n_samples, grid = u0.shape
    k = 2 * np.pi * np.fft.fftfreq(grid, d=1.0 / grid)
    ik = 1j * k
    lin = -nu * k ** 2  # linear operator in Fourier space

    # Dealiasing mask (2/3 rule).
    mask = np.abs(np.fft.fftfreq(grid, d=1.0 / grid)) < (grid / 3)

    # ETDRK2 coefficients (guard the k=0 singularities via a limit).
    E = np.exp(lin * dt)
    E2 = np.exp(lin * dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(np.abs(lin) > 1e-14, (E - 1.0) / lin, dt)
        f2 = np.where(np.abs(lin) > 1e-14, (E - 1.0 - lin * dt) / (lin ** 2 * dt), dt / 2.0)

    def nonlinear(u_hat: np.ndarray) -> np.ndarray:
        u = np.fft.ifft(u_hat, axis=-1).real
        flux = 0.5 * u ** 2
        return -(ik * mask) * np.fft.fft(flux, axis=-1)

    u_hat = np.fft.fft(u0, axis=-1)
    n_steps = int(round(t_final / dt))
    for _ in range(n_steps):
        N1 = nonlinear(u_hat)
        a = E * u_hat + f1 * N1
        N2 = nonlinear(a)
        u_hat = E * u_hat + f1 * N1 + f2 * (N2 - N1)

    return np.fft.ifft(u_hat, axis=-1).real


def downsample(u: np.ndarray, target_grid: int) -> np.ndarray:
    """Spectrally downsample the last axis to target_grid points."""
    grid = u.shape[-1]
    if target_grid == grid:
        return u
    if target_grid > grid:
        raise ValueError("target_grid must be <= current grid for downsampling")
    u_hat = np.fft.rfft(u, axis=-1)
    keep = target_grid // 2 + 1
    u_hat_small = u_hat[..., :keep] * (target_grid / grid)
    return np.fft.irfft(u_hat_small, n=target_grid, axis=-1)


def generate_dataset(n_samples: int = 1200, grid: int = 1024, nu: float = 0.01,
                     t_final: float = 1.0, dt: float = 1e-4, seed: int = 0):
    """Generate (u0, uT) pairs plus downsampled copies at 64/128/256 points."""
    rng = np.random.default_rng(seed)
    u0 = gaussian_random_field(n_samples, grid, rng=rng)
    uT = solve_burgers(u0, nu=nu, t_final=t_final, dt=dt)

    data = {"u0": u0.astype(np.float32), "uT": uT.astype(np.float32),
            "nu": nu, "t_final": t_final, "grid": grid}
    for g in (64, 128, 256):
        if g < grid:
            data[f"u0_{g}"] = downsample(u0, g).astype(np.float32)
            data[f"uT_{g}"] = downsample(uT, g).astype(np.float32)
    return data


def main():
    p = argparse.ArgumentParser(description="Generate a Burgers' dataset.")
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--grid", type=int, default=1024)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--t_final", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="data/burgers.npz")
    args = p.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data = generate_dataset(args.n_samples, args.grid, args.nu, args.t_final, args.dt, args.seed)
    np.savez_compressed(args.out, **data)
    print(f"Saved {args.n_samples} samples at grid {args.grid} to {args.out}")


if __name__ == "__main__":
    main()

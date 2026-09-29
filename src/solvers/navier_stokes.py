"""
2D incompressible Navier-Stokes solver (vorticity-streamfunction form).

Purpose
    Generate ground-truth data for the 2D FNO (Part 2). Evolve the vorticity
    field omega(x, y, t) forward in time on a periodic square, saving snapshots
    along each trajectory.

The equation (vorticity transport, periodic domain [0, 2*pi)^2)

    d omega/dt + (u . grad) omega = nu * laplacian(omega) + f

    omega      scalar vorticity  (omega = dv/dx - du/dy)
    u = (u, v) velocity, recovered from omega via the streamfunction
    nu         viscosity
    f          external forcing that keeps the flow from decaying

Velocity from vorticity, in Fourier space:
    Poisson:   psi_hat = omega_hat / k^2     (streamfunction; k=0 mode set to 0)
    velocity:  u = d psi/dy, v = -d psi/dx   (multiply by ik)

Method
    Pseudo-spectral in space (2D FFT for all derivatives, Concept 1),
    integrating factor for the viscous term (Concept 5), 2/3 dealiasing for the
    nonlinear advection (Concept 5). The same toolkit as Burgers, one dimension
    up.

Functions, in order
    1. spectral_grid(N): 2D wavenumbers kx, ky, k^2, and the dealias mask.
    2. vorticity_to_velocity(w_hat): solve Poisson, return (u, v).
    3. rhs(w_hat): advection plus forcing, in Fourier space.
    4. solve_ns(...): integrating-factor time stepping, save snapshots.
    5. generate_dataset(): many trajectories to a saved tensor (runs on GPU).

Data shape
    Each trajectory is a time series of vorticity snapshots, shape (T, N, N).
    The FNO learns to advance one step: given omega at time t, predict omega at
    t+1, applied autoregressively for a rollout.
"""

import numpy as np


def spectral_grid(N):
    """2D wavenumbers and the dealias mask for an N x N periodic grid."""
    k = np.fft.fftfreq(N, d=1.0 / N)      # integer wavenumbers, domain [0, 2*pi)
    kx = k.reshape(N, 1)                   # varies along x (axis 0)
    ky = k.reshape(1, N)                   # varies along y (axis 1)
    k2 = kx**2 + ky**2                     # |k|^2, used by the Laplacian / Poisson

    k2 = k2.copy()
    k2[0, 0] = 1.0                         # avoid divide-by-zero at k=0 in the Poisson solve

    # 2/3 dealias mask: keep the low modes in each direction.
    dealias = (np.abs(kx) < N / 3) & (np.abs(ky) < N / 3)
    return kx, ky, k2, dealias


def vorticity_to_velocity(w_hat, kx, ky, k2):
    """Recover velocity (u, v) in physical space from vorticity in Fourier space."""
    psi_hat = w_hat / k2                             # Poisson: psi_hat = w_hat / k^2
    u = np.fft.ifft2( 1j * ky * psi_hat).real        # u =  d psi/dy
    v = np.fft.ifft2(-1j * kx * psi_hat).real        # v = -d psi/dx
    return u, v

def rhs(w_hat, kx, ky, k2, dealias, forcing_hat):
    """Non-diffusion part in Fourier space:  N = -(u . grad) w + f."""
    u, v = vorticity_to_velocity(w_hat, kx, ky, k2)
    wx = np.fft.ifft2(1j * kx * w_hat).real       # d omega/dx
    wy = np.fft.ifft2(1j * ky * w_hat).real       # d omega/dy
    advection = u * wx + v * wy                    # (u . grad) omega, physical space
    N_hat = -np.fft.fft2(advection) * dealias      # dealiased nonlinear term
    return N_hat + forcing_hat


def solve_ns(w0, nu=1e-3, t_final=1.0, dt=1e-3, save_every=1, forcing_amp=0.1):
    """March vorticity forward, returning saved snapshots (n_saved, N, N)."""
    N = w0.shape[-1]
    kx, ky, k2, dealias = spectral_grid(N)

    # Integrating factor for the viscous term (Concept 5), precomputed once.
    E = np.exp(-nu * k2 * dt)

    # Fixed forcing (keeps the flow from decaying), built once in Fourier space.
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = forcing_amp * (np.sin(X + Y) + np.cos(X + Y))
    forcing_hat = np.fft.fft2(f)

    w_hat = np.fft.fft2(w0)
    n_steps = int(round(t_final / dt))
    snaps = []
    for step in range(n_steps):
        w_hat = E * (w_hat + dt * rhs(w_hat, kx, ky, k2, dealias, forcing_hat))
        if (step + 1) % save_every == 0:
            snaps.append(np.fft.ifft2(w_hat).real)
    return np.array(snaps)

def gaussian_random_field_2d(n_samples, N, alpha=2.0, tau=3.0, seed=None):
    """Smooth random initial vorticity fields, shape (n_samples, N, N)."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N, d=1.0 / N)
    kx = k.reshape(N, 1); ky = k.reshape(1, N); k2 = kx**2 + ky**2
    scale = (k2 + tau**2) ** (-alpha / 2); scale[0, 0] = 0.0
    noise = rng.standard_normal((n_samples, N, N)) + 1j * rng.standard_normal((n_samples, N, N))
    w = np.fft.ifft2(noise * scale).real
    peak = np.abs(w).reshape(n_samples, -1).max(axis=1).reshape(n_samples, 1, 1)
    return w / peak                                   # normalize each field to amplitude ~1


def generate_dataset(n_traj=100, N=64, nu=1e-3, dt=1e-3, t_final=10.0,
                     n_snapshots=20, seed=0):
    """Generate NS vorticity trajectories, shape (n_traj, n_snapshots, N, N)."""
    w0 = gaussian_random_field_2d(n_traj, N, seed=seed)
    n_steps = int(round(t_final / dt))
    save_every = max(1, n_steps // n_snapshots)
    snaps = solve_ns(w0, nu=nu, t_final=t_final, dt=dt, save_every=save_every)  # (n_saved, n_traj, N, N)
    return np.transpose(snaps, (1, 0, 2, 3)).astype(np.float32)                 # (n_traj, n_saved, N, N)
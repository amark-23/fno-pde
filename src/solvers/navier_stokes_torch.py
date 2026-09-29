"""
2D Navier-Stokes solver, PyTorch version (GPU-capable).

A faithful port of navier_stokes.py (NumPy) to torch.fft, so dataset generation
runs on a GPU. The NumPy module stays the readable reference; this one is for
scale. The math is identical (see THEORY.md Concept 13); only the array library
and the device change.
"""
import torch


def spectral_grid(N, device):
    """2D wavenumbers and dealias mask on `device`."""
    k = torch.fft.fftfreq(N, d=1.0 / N, device=device)   # integer wavenumbers, [0, 2*pi)
    kx = k.reshape(N, 1)
    ky = k.reshape(1, N)
    k2 = kx**2 + ky**2
    k2 = k2.clone(); k2[0, 0] = 1.0                        # avoid /0 in the Poisson solve
    dealias = ((kx.abs() < N / 3) & (ky.abs() < N / 3)).to(torch.float32)
    return kx, ky, k2, dealias


def vorticity_to_velocity(w_hat, kx, ky, k2):
    """Velocity (u, v) in physical space from vorticity in Fourier space."""
    psi_hat = w_hat / k2                                   # Poisson: psi_hat = w_hat / k^2
    u = torch.fft.ifft2( 1j * ky * psi_hat).real           # u =  d psi/dy
    v = torch.fft.ifft2(-1j * kx * psi_hat).real           # v = -d psi/dx
    return u, v


def rhs(w_hat, kx, ky, k2, dealias, forcing_hat):
    """Non-diffusion part in Fourier space:  N = -(u . grad) w + f."""
    u, v = vorticity_to_velocity(w_hat, kx, ky, k2)
    wx = torch.fft.ifft2(1j * kx * w_hat).real
    wy = torch.fft.ifft2(1j * ky * w_hat).real
    advection = u * wx + v * wy
    N_hat = -torch.fft.fft2(advection) * dealias
    return N_hat + forcing_hat


def solve_ns(w0, nu=1e-3, t_final=1.0, dt=1e-3, save_every=1, forcing_amp=0.1):
    """March vorticity forward; return snapshots stacked on a new leading axis."""
    device = w0.device
    N = w0.shape[-1]
    kx, ky, k2, dealias = spectral_grid(N, device)

    E = torch.exp(-nu * k2 * dt)                           # integrating factor

    xs = torch.linspace(0, 2 * torch.pi, N + 1, device=device)[:-1]
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    f = forcing_amp * (torch.sin(X + Y) + torch.cos(X + Y))
    forcing_hat = torch.fft.fft2(f)

    w_hat = torch.fft.fft2(w0)
    n_steps = int(round(t_final / dt))
    snaps = []
    for step in range(n_steps):
        w_hat = E * (w_hat + dt * rhs(w_hat, kx, ky, k2, dealias, forcing_hat))
        if (step + 1) % save_every == 0:
            snaps.append(torch.fft.ifft2(w_hat).real)
    return torch.stack(snaps)                              # (n_saved, ..., N, N)


def gaussian_random_field_2d(n_samples, N, device, alpha=2.0, tau=3.0, seed=None):
    """Smooth random initial vorticity fields on `device`, shape (n_samples, N, N)."""
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    k = torch.fft.fftfreq(N, d=1.0 / N, device=device)
    kx = k.reshape(N, 1); ky = k.reshape(1, N); k2 = kx**2 + ky**2
    scale = (k2 + tau**2) ** (-alpha / 2); scale[0, 0] = 0.0
    re = torch.randn(n_samples, N, N, generator=g, device=device)
    im = torch.randn(n_samples, N, N, generator=g, device=device)
    w = torch.fft.ifft2((re + 1j * im) * scale).real
    peak = w.abs().reshape(n_samples, -1).amax(dim=1).reshape(n_samples, 1, 1)
    return w / peak


def generate_dataset(n_traj=100, N=64, nu=1e-3, dt=1e-3, t_final=10.0,
                     n_snapshots=20, seed=0, device=None):
    """Generate NS vorticity trajectories, returned as a NumPy array
    (n_traj, n_snapshots, N, N). Runs on GPU when available."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    w0 = gaussian_random_field_2d(n_traj, N, device, seed=seed)
    n_steps = int(round(t_final / dt))
    save_every = max(1, n_steps // n_snapshots)
    snaps = solve_ns(w0, nu=nu, t_final=t_final, dt=dt, save_every=save_every)  # (n_saved, n_traj, N, N)
    data = snaps.permute(1, 0, 2, 3).contiguous()                              # (n_traj, n_saved, N, N)
    return data.cpu().numpy().astype("float32")

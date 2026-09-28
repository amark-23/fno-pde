"""
1D viscous Burgers' equation solver.

PURPOSE
-------
Generate ground-truth data for the FNO. Given an initial field u(x, 0), compute
the field u(x, 1) at a later time by simulating the physics.

THE EQUATION (viscous Burgers', periodic domain x in [0, 1))

    u_t + u * u_x = nu * u_xx

    u        the field we evolve over time    (a velocity profile)
    u_t      how u changes in time            (what we step forward)
    u * u_x  nonlinear advection              (steepens the profile -> shocks)
    nu * u_xx  viscous diffusion              (smooths out sharp gradients)
    nu       viscosity: small nu -> sharper shocks, large nu -> smoother

METHOD
------
Pseudo-spectral in space: use the FFT to compute the spatial derivatives
u_x and u_xx (Concept #1 in THEORY.md -> derivative = multiply by ik).
Then march forward in time in many small steps dt, from t = 0 to t = t_final.

PLAN — functions we'll build, in order
--------------------------------------
1. gaussian_random_field(...)   sample smooth, random initial conditions u(x, 0)
2. solve_burgers(...)           march one u0 forward in time to u(x, t_final)
3. (later) downsample(...)      make coarser-grid copies (for resolution tests)
4. (later) generate_dataset()   + main(): produce and save many (u0, uT) pairs

OPEN DECISION (before we write solve_burgers)
---------------------------------------------
How to step forward in time. Options range from a simple explicit scheme to a
"stiff-aware" one that handles the nu * u_xx term more robustly. We'll pick this
together and note the reasoning in THEORY.md.
"""

import numpy as np


def gaussian_random_field(n_samples, grid, alpha=2.5, tau=7.0, seed=None):
    """Sample smooth, periodic initial conditions u(x, 0).

    Why random: we want the FNO to learn the mapping for a whole *distribution*
    of inputs, not one specific initial condition.
    Why smooth: real fields aren't white noise; we bias toward low frequencies.
    """
    # 1. Wavenumbers: which frequencies the grid can represent.
    k = np.fft.fftfreq(grid, d=1.0 / grid)      # -> [0, 1, 2, ..., -2, -1]

    # 2. Decaying scale factor s(k): crush high frequencies
    scale = (tau ** (alpha - 1)) * (tau ** 2 + (2 * np.pi * k) ** 2) ** (-alpha / 2)
    scale[0] = 0.0                               # drop the k=0 (mean) mode
    
    # 3. Random complex coefficients, one per (sample, wavenumber).
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal((n_samples, grid)) + 1j * rng.standard_normal((n_samples, grid))

    # 4. Shape the spectrum, then transform back to physical space.
    coeffs = coeffs * scale[None, :] * grid      # apply s(k) to every sample
    field = np.fft.ifft(coeffs, axis=-1).real    # -> real curve u(x, 0)

    return field    


def solve_burgers(u0, nu=0.01, t_final=1.0, dt=1e-4):
    """March an initial field u0 forward in time to u(x, t_final).

    Uses the FFT for the spatial derivatives (u_x, u_xx). The time-stepping
    scheme is the open decision above.
    """
    n_samples, grid = u0.shape

    # Derivative operators in Fourier space (Concept #1).
    k  = 2 * np.pi * np.fft.fftfreq(grid, d=1.0 / grid)   # angular wavenumbers
    ik = 1j * k          # d/dx      -> multiply by ik
    k2 = k ** 2          # d^2/dx^2  -> multiply by -k^2

    def rhs(u):
        """Rate of change  F(u) = -u * u_x + nu * u_xx."""
        u_hat = np.fft.fft(u, axis=-1)                 # to Fourier space
        u_x  = np.fft.ifft(ik * u_hat, axis=-1).real   # first derivative
        u_xx = np.fft.ifft(-k2 * u_hat, axis=-1).real  # second derivative
        return -u * u_x + nu * u_xx    
    
    n_steps = int(round(t_final / dt))
    u = u0.copy()                    # don't overwrite the caller's array
    for _ in range(n_steps):
        u = u + dt * rhs(u)          # Euler step: u_next = u_now + dt * F(u)
    return u
    
def solve_burgers_if(u0, nu=0.01, t_final=1.0, dt=1e-3): 
    """March a field u0 forward in time to u(x, t_final) using robust scheme."""
    n_samples, grid = u0.shape

    # Wavenumbers and first-derivative operator (as before).
    m  = np.fft.fftfreq(grid, d=1.0 / grid)   # integer wavenumbers
    k  = 2 * np.pi * m
    ik = 1j * k

    # NEW #1: the integrating factor — the exact linear (diffusion) solve.
    E = np.exp(-nu * k**2 * dt)               # always in (0, 1]; never blows up

    # NEW #2: 2/3 dealiasing mask — keep low 2/3 of modes, zero the top third.
    dealias = np.abs(m) < grid / 3

    def nonlinear_hat(u):
        # N = -u * u_x, returned in Fourier space and dealiased.
        u_hat = np.fft.fft(u, axis=-1)
        u_x   = np.fft.ifft(ik * u_hat, axis=-1).real
        N_hat = np.fft.fft(-u * u_x, axis=-1)
        return N_hat * dealias          # zero the top third of modes

    # Integrating-factor Euler loop.
    n_steps = int(round(t_final / dt))
    u = u0.copy()
    for _ in range(n_steps):
        u_hat = np.fft.fft(u, axis=-1)
        u_hat = E * (u_hat + dt * nonlinear_hat(u))   
        u = np.fft.ifft(u_hat, axis=-1).real
    return u


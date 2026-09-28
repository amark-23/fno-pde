"""
1D viscous Burgers' equation solver.

PURPOSE
-------
Generate ground-truth data for the FNO. Given an initial field u(x, 0), compute
the field u(x, 1) at a later time by simulating the physics. This is the
"teacher" (see THEORY.md > Big picture). Pure NumPy: no ML, no gradients.

THE EQUATION (viscous Burgers', periodic domain x in [0, 1))

    u_t + u * u_x = nu * u_xx

    u        the field we evolve over time (think: a velocity profile)
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


def gaussian_random_field():
    """Sample smooth, periodic initial conditions u(x, 0).

    Why random: we want the FNO to learn the mapping for a whole *distribution*
    of inputs, not one specific initial condition.
    Why smooth: real fields aren't white noise; we bias toward low frequencies.

    TODO: build together. (args and body to come)
    """
    ...


def solve_burgers():
    """March an initial field u0 forward in time to u(x, t_final).

    Uses the FFT for the spatial derivatives (u_x, u_xx). The time-stepping
    scheme is the open decision above.

    TODO: build together. (args and body to come)
    """
    ...

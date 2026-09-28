# Theoretical Background

Notes on the concepts and formulas behind this project, so anyone reading the
repository can follow *what* is being computed and *why*. Informal for now;
we tighten it up later. Entries are added as we encounter each idea.

---

## Big picture — what this project is

There are **two separate machines**, and it's easy to fuse them because the FFT
appears in both. Keep them apart:

- **The solver (classical physics, no ML).** Computes the true final field
  u(x, 1) from an initial field u(x, 0) by marching forward through many small
  time steps, using the FFT at each step to compute spatial derivatives. This is
  the **teacher**: it generates ground-truth data.
- **The FNO (the ML).** A neural network trained by backpropagation to imitate
  that same mapping in one shot. This is the **student**.

**Through-line of the project:**

> solver makes the data → FNO learns to replace the solver → compare them on
> accuracy, speed, and generalization.

**In supervised-learning terms** (the skeleton you already know):

- input X = u(x, 0), target y = u(x, 1)
- training set = the (input, target) pairs the solver produces
- loss = relative L2 between prediction and the solver's answer
- optimizer = Adam + backprop

What makes it *modern ML research* rather than a toy: the inputs and outputs are
whole **functions**, not feature vectors (this is called **operator learning**),
and once trained the network is far faster than the solver and generalizes to
grid resolutions it never saw. That places it in **scientific ML / "AI for
science."**

**Two one-liners worth remembering:**

> *Teacher vs student.* The solver **computes** u(x, 1) to near-exact accuracy;
> the FNO **approximates** that mapping with a fast learned shortcut, and we
> measure how close it gets.

> *What the solver does.* It marches u(x, 0) forward through many small time
> steps to reach u(x, 1), using the FFT at each step to compute the spatial
> derivatives it needs. Two ingredients: stepping through time (what gets you
> there) + FFT for derivatives (what makes each step cheap and accurate).

Backpropagation lives only in the FNO. The solver never sees a gradient.

---

## 1. Differentiation becomes multiplication in Fourier space

**The idea.** Taking a derivative in physical space is the same as multiplying
each frequency component by `ik` in Fourier space. This is what lets us compute
spatial derivatives exactly (up to the resolved modes) instead of approximating
them with finite differences — the basis of the *spectral* method.

**Why it's true.** The Fourier transform writes a function as a sum of complex
exponentials:

$$f(x) = \int \hat{f}(k) e^{i 2\pi k x} \mathrm{d}k$$

Each building block is $e^{i 2\pi k x}$. By Euler's formula
$e^{i\theta} = \cos\theta + i\sin\theta$, so these are just sines and cosines.
Differentiating one mode brings the frequency down as a factor:

$$\frac{d}{dx} e^{i 2\pi k x} = (i 2\pi k) \cdot e^{i 2\pi k x}$$

Since differentiation is linear, doing it to the whole sum multiplies every
mode by its own $i 2\pi k$. In transform terms:

$$\widehat{f'}(k) = (i 2\pi k) \cdot \hat{f}(k)$$

Second derivative: multiply by $(i 2\pi k)^2 = -(2\pi k)^2$.

**How the solver uses it.** To get $u_x$ (or $u_{xx}$): FFT the field, multiply
by $ik$ (or $-k^2$), inverse-FFT back. In the code, the wavenumbers already
absorb the $2\pi$:

```
k  = 2*pi*fftfreq(grid, d=1/grid)   # wavenumbers
ik = 1j*k                            # derivative operator in Fourier space
```

This is exact for every frequency the grid can represent, which is why spectral
derivatives are far more accurate than finite differences at the same
resolution.

**Caveat (for later).** This clean rule assumes the function is periodic on the
domain. Our domain is $[0, 1)$ with periodic boundaries, so it applies. Products
of fields (like the nonlinear term $u \cdot u_x$) introduce new frequencies and
need care — we'll note that when we hit dealiasing.

---

## 2. The Burgers' equation — what each term does

The PDE we simulate (viscous Burgers', periodic domain $x \in [0, 1)$):

$$u_t + u \cdot u_x = \nu \cdot u_{xx}$$

Reading the terms:

- $u_t$ — how the field changes in time. **This is what we step forward.**
- $u \cdot u_x$ — **nonlinear advection.** The field transports itself, and
  larger values move faster, so the profile piles up and *steepens*. This term
  is what creates shocks.
- $\nu \cdot u_{xx}$ — **viscous diffusion.** Smooths sharp gradients, spreading
  them back out.
- $\nu$ — **viscosity.** Small $\nu$: steepening wins, sharp shocks. Large
  $\nu$: diffusion wins, smooth fields.

**The intuition — a tug-of-war.** Advection steepens, diffusion smooths. A
*shock* is what you see when steepening temporarily wins: the profile sharpens
toward a near-discontinuity, until diffusion catches up and caps how sharp it
can get. The value of $\nu$ sets where that balance lands.

**Why it's a good test problem.** It's the simplest PDE that has *both*
nonlinearity ($u \cdot u_x$) and diffusion ($\nu \cdot u_{xx}$), so it exercises
the hard parts of a solver — and later, of the FNO — without the full
complexity of something like Navier–Stokes.

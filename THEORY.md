# Theoretical Background

Notes on the concepts and formulas behind this project, so anyone reading the
repository can follow *what* is being computed and *why*. Informal for now;
we tighten it up later. Entries are added as we encounter each idea.

---

## 1. Differentiation becomes multiplication in Fourier space

**The idea.** Taking a derivative in physical space is the same as multiplying
each frequency component by `ik` in Fourier space. This is what lets us compute
spatial derivatives exactly (up to the resolved modes) instead of approximating
them with finite differences — the basis of the *spectral* method.

**Why it's true.** The Fourier transform writes a function as a sum of complex
exponentials:

$$f(x) = \int \hat{f}(k)\, e^{i 2\pi k x}\, dk$$

Each building block is $e^{i 2\pi k x}$. By Euler's formula
$e^{i\theta} = \cos\theta + i\sin\theta$, so these are just sines and cosines.
Differentiating one mode brings the frequency down as a factor:

$$\frac{d}{dx}\, e^{i 2\pi k x} = i 2\pi k \; e^{i 2\pi k x}$$

Since differentiation is linear, doing it to the whole sum multiplies every
mode by its own $i 2\pi k$. In transform terms:

$$\widehat{f'}(k) = i 2\pi k \; \hat{f}(k)$$

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
of fields (like the nonlinear term $u\,u_x$) introduce new frequencies and need
care — we'll note that when we hit dealiasing.

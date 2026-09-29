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

---

## 3. Sampling initial conditions — the Gaussian random field

**What and why.** We need many *different* smooth starting fields u(x, 0) so the
FNO learns the mapping for a whole distribution of inputs, not one curve. A
**Gaussian random field (GRF)** lets us draw a random *function* (a random
curve); any finite set of its values is jointly Gaussian.

**Smoothness = suppressing high frequencies.** We build the field in Fourier
space (using Concept #1):

1. Give each wavenumber $k$ a **random** complex coefficient (real and imaginary
   parts drawn from a standard normal).
2. Multiply by a **scale factor that decays with $k$**:

$$s(k) = \tau^{\alpha - 1} \cdot (\tau^2 + (2\pi k)^2)^{-\alpha/2}$$

3. **Inverse-FFT** back to a real curve, and drop the $k = 0$ mode so the field
   is centered around zero.

Because high-frequency modes get small amplitudes, the curve comes out smooth.

**The knobs.**

- $\alpha$ — the main **smoothness** dial. Larger $\alpha$ makes $s(k)$ fall off
  faster as $k$ grows, so high frequencies are crushed harder and the curve is
  smoother.
- $\tau$ — a length-scale knob; usually fixed. Sets the overall balance of low-
  vs high-frequency content.

**Why "Gaussian."** The coefficients are Gaussian, so the field values (linear
combinations of them) are Gaussian too — the simplest random model, fully
described by its mean and covariance. The decaying spectrum $s(k)^2$ is just the
Fourier-space form of that covariance.

---

## 4. Marching forward in time (time-stepping)

**The rate of change.** Rearrange Burgers' to isolate the time derivative:

$$u_t = -u \cdot u_x + \nu \cdot u_{xx}$$

Call the right-hand side $F(u)$ — it tells us how fast $u$ is changing right now.

**The Euler step.** To advance a small step $dt$:

$$u_{\text{next}} = u_{\text{now}} + dt \cdot F(u_{\text{now}})$$

Repeat this many times, from $t = 0$ to $t = t_{\text{final}}$. That is the whole
solver loop.

**Analogy to optimization.** This is structurally like a gradient-descent step:
instead of nudging parameters downhill by a gradient with a learning rate, we
nudge the field forward in time by $F(u)$ with a step size $dt$. And $dt$ behaves
like a learning rate — **too large and the recursion blows up** (values race off
to infinity / NaN), too small and it is just slow.

**Stiffness — why $dt$ can be forced tiny.** The diffusion term $\nu \cdot u_{xx}$
becomes $-\nu (2\pi k)^2 \cdot \hat{u}$ in Fourier space. For high wavenumbers $k$
that is a large negative number, and an explicit step is only stable when

$$dt \lesssim \frac{1}{\nu \cdot k_{\max}^2}$$

Since $k_{\max}$ grows with the grid size, high-resolution grids force very small
steps. This is called **stiffness**.

**Two schemes.**

- **Explicit** (simple): just use a small-enough $dt$. Easy to write and
  understand; the cost is many small steps.
- **Stiff-aware / robust**: treat the linear diffusion term *exactly* in Fourier
  space, which removes the stability limit and allows much larger $dt$. A little
  more theory and code.

**Our choice.** Start with the **explicit** scheme to learn the mechanics and
watch a shock form (and, on purpose, watch it blow up when $dt$ is too big).
Then upgrade to the **robust** scheme, because one of our experiments
(resolution transfer) needs ground-truth data at grid 1024, where the explicit
$dt$ limit would otherwise make data generation painfully slow.

---

## 5. The robust stepper — integrating factor + dealiasing

**Problem recap.** The explicit scheme's $dt$ was limited by the *stiff* linear
term $-\nu k^2 \hat{u}$: for high $k$ it is huge, forcing tiny steps. The
nonlinear term was never the bottleneck.

**Split each Fourier mode into linear + nonlinear.**

$$\frac{d\hat{u}}{dt} = \underbrace{-\nu k^2 \hat{u}}_{\text{linear, stiff}} + \underbrace{\hat{N}}_{\text{nonlinear, mild}}, \qquad \hat{N} = \text{FFT}(-u \cdot u_x)$$

**Solve the linear part exactly.** On its own, $d\hat{u}/dt = -\nu k^2 \hat{u}$
has the closed form $\hat{u}(t+dt) = e^{-\nu k^2 dt} \cdot \hat{u}(t)$. That
factor is **always in $(0, 1]$**: it equals 1 at $k = 0$ and $\to 0$ as
$k \to \infty$. Because it is never above 1, multiplying by it can only shrink a
mode — it can **never blow up**, for any $k$ or $dt$. (Contrast the explicit
scheme, which multiplied a mode by $(1 - \nu k^2 dt)$; for large $k \cdot dt$
that exceeds 1 in magnitude and amplifies the mode → explosion.)

**Integrating-factor update.** Treat the stiff linear part with its exact
exponential; step only the mild nonlinear part explicitly:

$$\hat{u}_{\text{next}} = e^{-\nu k^2 dt} \cdot \left( \hat{u}_{\text{now}} + dt \cdot \hat{N}_{\text{now}} \right)$$

The stiff stability limit is gone; only the mild advective step-size limit from
the nonlinear term remains, so $dt$ can be much larger.

**Dealiasing (the 2/3 rule).** The product $u \cdot u_x$ creates frequencies
higher than the grid can represent; they fold back as false low-frequency noise
("aliasing") that can corrupt or destabilize the solution. Fix: zero out the top
third of wavenumbers in the nonlinear term before using it. Cheap, and it
removes the aliasing.

**Result.** Integrating factor (tames stiffness) + 2/3 dealiasing (tames the
product) → a solver that stays stable at large $dt$ and high resolution, so
grid-1024 data generation is fast. We keep the explicit `solve_burgers` for
comparison and add this as `solve_burgers_if`.

---

## 6. Verifying the solver — the convergence check

**The problem.** We treat the solver's output as the true $u(x, 1)$, but every
time-stepper carries an error that depends on $dt$ (for our first-order scheme,
error $\propto dt$). We have no exact solution to compare against.

**The test: self-consistency.** Instead of comparing to truth, check whether the
answer *stops changing* as $dt$ shrinks. Solve at $dt, dt/2, dt/4, \dots$, take
the finest as a reference, and measure the relative L2 error of the coarser runs
against it. If refining $dt$ barely changes the result, the time error is already
negligible and the solver is converged — trustworthy as ground truth.

**Result (our IF solver, grid 256, $\nu = 0.01$).**

| $dt$ | rel-L2 vs finest | change when $dt$ halved |
| --- | --- | --- |
| 2.0e-3 | 2.2e-3 | 1.2e-3 |
| 1.0e-3 | 1.0e-3 | 5.9e-4 |
| 5.0e-4 | 4.4e-4 | 2.9e-4 |
| 2.5e-4 | 1.5e-4 | 1.5e-4 |

At our working $dt = 10^{-3}$, halving the step changes the answer by only
$\approx 6 \times 10^{-4}$ (0.06%) — the "halving barely changes it" criterion is
met.

**Two things this confirms.**

1. **Converged** — the error is tiny at our working $dt$, so the data is
   trustworthy.
2. **Correct order** — each halving of $dt$ roughly halves the error, the
   signature of a first-order scheme (a straight line of slope $\approx 1$ on a
   log-log plot). Erratic or stalled error would instead hint at a bug, so this
   clean trend is also a sanity check on the implementation.

---

## 7. Downsampling — the same solution at several resolutions

**Why we need it.** The FNO's headline property is that it is
*resolution-invariant*: one trained model can run on any grid size. The
resolution-transfer experiment tests this by training at a coarse grid (e.g. 64)
and evaluating at finer ones (128, 256, 1024). To make that comparison fair, we
need the **same** solutions represented at each resolution. So we generate the
data once at a high base resolution, then downsample $u_0$ and $u_T$ to the
coarser grids — the coarse versions are exact restrictions of the same fields,
not separately generated ones.

**How — spectral (Fourier) downsampling.** Naively keeping every $k$-th grid
point can alias and distort. Instead: FFT the field, **keep only the lowest
wavenumbers the coarse grid can represent** (up to its Nyquist limit), and
inverse-FFT at the smaller size. Because a smooth field's energy lives in its low
modes, dropping the high ones is nearly lossless — this is the natural way to
represent a band-limited function on fewer points. A scale factor
(target / source) corrects the FFT normalization so amplitudes are preserved.

---

## 8. The spectral convolution — the heart of the FNO

**The move.** Same FFT hop as the solver (Concept #1), but the multiplier is
**learned** instead of the fixed $ik$:

1. FFT the input field to Fourier space.
2. Keep only the lowest `modes` coefficients; discard the high ones.
3. Multiply each kept coefficient by a **learned complex weight**.
4. Inverse-FFT back to a field.

$$(\mathcal{K}v)(x) = \mathcal{F}^{-1}\big(R \cdot \mathcal{F}(v)\big)(x)$$

where $R$ are the learned per-mode weights (a small complex tensor).

**What is actually learned.** Not a kernel in physical space — we parametrize
the filter *directly by its Fourier coefficients* $R$ (a handful of complex
numbers). The FFT does not "discover" a kernel; it just carries the signal into
and out of the space where the parameters live. By the convolution theorem, a
multiply in Fourier space equals a convolution in physical space, so $R$
implicitly defines a **global** convolution filter — one spanning the whole
domain, unlike a CNN's small local kernel. That global reach is why the FNO
captures long-range structure cheaply.

**Why keep only low modes.**

- *Regularization / efficiency.* The solution's important structure lives in the
  low frequencies (same intuition as the GRF and downsampling), so learning
  weights for just those is enough and generalizes better.
- *Resolution independence.* "The lowest 16 modes" means the same physical
  frequencies (0, 1, ..., 15 cycles across the domain) on any grid — a grid of
  64 or 1024 both contain them. So a layer that learns 16 complex weights plugs
  into any resolution unchanged. This is the crux of the resolution-transfer
  property.

**Contrast with the solver.** Solver: FFT → multiply by fixed $ik$ → iFFT (a
known derivative). FNO: FFT → multiply by learned $R$ → iFFT (a learned
operator). Same machinery; fixed vs trainable multiplier.

---

## 9. The full FNO architecture

The spectral conv (Concept #8) is the core piece; the model wraps it in a
standard lift -> process -> project structure.

**Pipeline.** Input $u(x, 0)$, shape (batch, grid, in_channels):

1. **Lift** — a `Linear` layer raises `in_channels` to a wider hidden dimension
   `width` (e.g. 64), giving the network room to represent features. Input has
   `in_channels = 2`: the field plus its x-coordinate (feeding the grid
   coordinate in is a small standard trick that helps the model).
2. **Fourier layers** ($\times$ depth, e.g. 4) — each runs **two paths in
   parallel and adds them**, then applies a GELU nonlinearity:
   - *Spectral conv* — global, but low-frequency only (it keeps only the low
     modes).
   - *Pointwise 1x1 conv* — local (acts at each grid point independently), but
     carries **all** frequencies, including the high ones the spectral path
     discarded.

   $$x \leftarrow \text{GELU}\big(\text{SpectralConv}(x) + W x\big)$$

3. **Project** — two `Linear` layers bring `width` down to `out_channels` (the
   predicted field), with a GELU between.

Output: predicted $u(x, 1)$, shape (batch, grid, out_channels).

**Why both paths.** The spectral conv is global but drops high modes; the 1x1
conv is local but keeps them. Added together they cover both **global
low-frequency structure** and **local high-frequency detail** — neither alone is
enough. This matters for shocks especially, which are sharp (high-frequency)
features the spectral path throws away and the local path preserves.

**Why the nonlinearity.** Without the GELU, stacking layers would collapse to a
single linear operator. The nonlinearity between layers is what lets the model
represent a nonlinear solution operator (Burgers' is nonlinear).

**Implementation note.** `Linear` layers expect channels last
(batch, grid, width); the conv-style layers expect channels in the middle
(batch, width, grid). The forward pass permutes between the two conventions —
bookkeeping, not math.

---

## 10. Training the FNO

Standard supervised learning — the FNO (student) learns to imitate the solver
(teacher). A few choices are specific to operator learning.

**The supervised setup.**

- Input $X = [u_0, x\text{-coordinate}]$ (2 channels); target $y = u_T$.
- Loss = **relative L2** (below); optimizer = Adam with weight decay; a cosine
  learning-rate schedule; batches via a `DataLoader`.

**Relative L2 loss.** Per sample $\lVert \text{pred} - \text{target}\rVert /
\lVert \text{target}\rVert$, then averaged over the batch.

- *Why not plain MSE.* Relative L2 normalizes each sample by its own magnitude,
  so large- and small-amplitude fields contribute equally; MSE would let
  large-magnitude samples dominate.
- *Scale-invariance.* Comparable across samples, resolutions and datasets — the
  standard FNO metric, and the same one used to verify the solver (Concept #6).

**Normalization.** Standardize the field channel (and the targets) to zero mean
/ unit variance using **training** statistics, so the optimizer sees well-scaled
inputs. Predictions are **decoded** back to physical units before the error is
measured, so the reported number matches the solver's real output.

**Result (grid 256, 1000 train, 100 epochs).** Test relative L2 $\approx 1.6
\times 10^{-3}$ (about 0.16%) — the FNO reproduces the solver to a fraction of a
percent, matching the original FNO paper's Burgers' result. Random weights would
give $\approx 1.0$ (100% error).

**Config-driven.** All settings (grid, modes, width, depth, epochs, lr, training
size) live in a YAML config, so the upcoming experiments — baseline comparison,
resolution transfer, data efficiency — are just different configs, not new code.

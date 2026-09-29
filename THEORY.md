# Theoretical background

These notes record the concepts and formulas behind the project so that a reader
of the repository can follow what is computed and why. Each entry corresponds to
one idea, in the order the code introduces it.

---

## Overview

Two distinct systems appear in this project, and because the FFT is used in both
they are easily confused. They must be kept separate.

The first is the solver: classical physics, with no machine learning. It computes
the true final field u(x, 1) from an initial field u(x, 0) by marching forward
through many small time steps, using the FFT at each step to compute spatial
derivatives. It acts as the teacher, generating ground-truth data.

The second is the FNO: the machine-learning model. It is a neural network trained
by backpropagation to reproduce that same mapping in a single step. It acts as
the student.

The project proceeds in three stages. The solver produces the data, the FNO
learns to replace the solver, and the two are compared on accuracy, speed, and
generalization. In the terms of supervised learning, the input is X = u(x, 0),
the target is y = u(x, 1), the training set is the collection of (input, target)
pairs the solver produces, the loss is the relative L2 error between prediction
and the solver's answer, and the optimizer is Adam with backpropagation.

What distinguishes this from ordinary regression is that the inputs and outputs
are whole functions rather than fixed-size feature vectors, a setting known as
operator learning. Once trained, the network is far faster than the solver and
generalizes to grid resolutions never seen during training, which places the work
within scientific machine learning.

Two summaries are worth keeping in mind. On the relationship between the systems,
the solver computes u(x, 1) to near-exact accuracy while the FNO approximates
that mapping with a fast learned shortcut, and the experiments measure how close
the approximation is. On what the solver does, it marches u(x, 0) forward through
many small time steps to reach u(x, 1), using the FFT at each step to compute the
spatial derivatives it needs, so that time-stepping advances the solution and the
FFT makes each step accurate and inexpensive. Backpropagation occurs only in the
FNO; the solver never computes a gradient.

---

## 1. Differentiation becomes multiplication in Fourier space

Taking a derivative in physical space is equivalent to multiplying each frequency
component by ik in Fourier space. This allows spatial derivatives to be computed
exactly, up to the resolved modes, rather than approximated by finite
differences, and it is the basis of the spectral method.

The reason follows from the Fourier transform, which writes a function as a sum
of complex exponentials:

$$f(x) = \int \hat{f}(k) e^{i 2\pi k x} \mathrm{d}k$$

Each building block is $e^{i 2\pi k x}$, and by Euler's formula
$e^{i\theta} = \cos\theta + i\sin\theta$ these are sines and cosines.
Differentiating one mode brings its frequency down as a factor:

$$\frac{d}{dx} e^{i 2\pi k x} = (i 2\pi k) \cdot e^{i 2\pi k x}$$

Because differentiation is linear, applying it to the whole sum multiplies every
mode by its own $i 2\pi k$:

$$\widehat{f'}(k) = (i 2\pi k) \cdot \hat{f}(k)$$

The second derivative multiplies by $(i 2\pi k)^2 = -(2\pi k)^2$.

In the solver, obtaining $u_x$ or $u_{xx}$ means transforming the field with the
FFT, multiplying by $ik$ or by $-k^2$, and transforming back. In the code the
wavenumbers already absorb the factor of $2\pi$:

```
k  = 2*pi*fftfreq(grid, d=1/grid)   # wavenumbers
ik = 1j*k                            # derivative operator in Fourier space
```

The result is exact for every frequency the grid can represent, which is why
spectral derivatives are far more accurate than finite differences at the same
resolution. The rule assumes the function is periodic on the domain; the domain
here is $[0, 1)$ with periodic boundaries, so it applies. Products of fields, such
as the nonlinear term $u \cdot u_x$, introduce new frequencies and require the
dealiasing treatment described in Concept 5.

---

## 2. The Burgers equation

The equation simulated is the viscous Burgers equation on the periodic domain
$x \in [0, 1)$:

$$u_t + u \cdot u_x = \nu \cdot u_{xx}$$

The term $u_t$ is the rate of change of the field in time, and it is the quantity
stepped forward. The term $u \cdot u_x$ is nonlinear advection: the field
transports itself, larger values move faster, and the profile piles up and
steepens, which is what creates shocks. The term $\nu \cdot u_{xx}$ is viscous
diffusion, which smooths sharp gradients and spreads them out. The parameter
$\nu$ is the viscosity: a small $\nu$ lets steepening win and produces sharp
shocks, while a large $\nu$ lets diffusion win and produces smooth fields.

The behavior is a competition between advection, which steepens, and diffusion,
which smooths. A shock appears when steepening temporarily dominates: the profile
sharpens toward a near-discontinuity until diffusion catches up and limits how
sharp it can become. The value of $\nu$ determines where that balance settles.

Burgers is a useful test problem because it is the simplest equation containing
both nonlinearity, through $u \cdot u_x$, and diffusion, through
$\nu \cdot u_{xx}$. It therefore exercises the difficult parts of both the solver
and the FNO without the full complexity of an equation such as Navier-Stokes.

---

## 3. Sampling initial conditions: the Gaussian random field

The FNO must learn the mapping for a whole distribution of inputs rather than a
single curve, so many different smooth initial fields u(x, 0) are required. A
Gaussian random field provides a way to draw a random function, that is a random
curve, for which any finite set of values is jointly Gaussian.

Smoothness corresponds to suppressing high frequencies, and the field is built in
Fourier space using Concept 1:

1. Each wavenumber $k$ is given a random complex coefficient, with real and
   imaginary parts drawn from a standard normal distribution.
2. The coefficients are multiplied by a scale factor that decays with $k$:

$$s(k) = \tau^{\alpha - 1} \cdot (\tau^2 + (2\pi k)^2)^{-\alpha/2}$$

3. The result is transformed back to a real curve, and the $k = 0$ mode is set to
   zero so the field is centered around zero.

Because high-frequency modes receive small amplitudes, the resulting curve is
smooth. The exponent $\alpha$ is the principal smoothness control: a larger
$\alpha$ makes $s(k)$ decay faster with $k$, suppressing high frequencies more
strongly and producing a smoother curve. The parameter $\tau$ sets an overall
length scale and the balance between low- and high-frequency content, and is
usually held fixed.

The name Gaussian refers to the coefficients being Gaussian, so the field values,
which are linear combinations of them, are Gaussian as well. This is the simplest
random model, fully described by its mean and covariance, and the decaying
spectrum $s(k)^2$ is the Fourier-space form of that covariance.

---

## 4. Marching forward in time

Rearranging Burgers to isolate the time derivative gives

$$u_t = -u \cdot u_x + \nu \cdot u_{xx}$$

The right-hand side is written $F(u)$ and gives the instantaneous rate of change
of $u$. Advancing by a small step $dt$ uses

$$u_{\text{next}} = u_{\text{now}} + dt \cdot F(u_{\text{now}})$$

repeated from $t = 0$ to $t = t_{\text{final}}$, which constitutes the entire
solver loop.

The update is structurally similar to a gradient-descent step: rather than moving
parameters downhill by a gradient scaled by a learning rate, the field is moved
forward in time by $F(u)$ scaled by a step size $dt$. The step size behaves like a
learning rate, in that too large a value causes the iteration to diverge to
infinity or NaN, and too small a value merely makes the computation slow.

The diffusion term $\nu \cdot u_{xx}$ becomes $-\nu (2\pi k)^2 \cdot \hat{u}$ in
Fourier space. For high wavenumbers this is a large negative number, and an
explicit step is stable only when

$$dt \lesssim \frac{1}{\nu \cdot k_{\max}^2}$$

Since $k_{\max}$ grows with grid size, high-resolution grids force very small
steps. This sensitivity is called stiffness.

Two schemes address it. The explicit scheme simply uses a sufficiently small
$dt$; it is easy to write and understand, at the cost of many small steps. The
stiff-aware, or robust, scheme treats the linear diffusion term exactly in
Fourier space, removing the stability limit and permitting a much larger $dt$ at
the cost of a little more theory and code. The project implements the explicit
scheme first, to establish the mechanics and to observe both shock formation and,
deliberately, divergence when $dt$ is too large, and then upgrades to the robust
scheme, which is needed because the resolution-transfer experiment requires
ground-truth data at grid 1024, where the explicit step-size limit would make
data generation prohibitively slow.

---

## 5. The robust stepper: integrating factor and dealiasing

The explicit step size was limited by the stiff linear term $-\nu k^2 \hat{u}$,
which is large for high $k$ and forces tiny steps; the nonlinear term was never
the bottleneck. The remedy begins by splitting each Fourier mode into a stiff
linear part and a mild nonlinear part:

$$\frac{d\hat{u}}{dt} = \underbrace{-\nu k^2 \hat{u}}_{\text{linear, stiff}} + \underbrace{\hat{N}}_{\text{nonlinear, mild}}, \qquad \hat{N} = \text{FFT}(-u \cdot u_x)$$

On its own, $d\hat{u}/dt = -\nu k^2 \hat{u}$ has the closed form
$\hat{u}(t+dt) = e^{-\nu k^2 dt} \cdot \hat{u}(t)$. This factor always lies in
$(0, 1]$: it equals 1 at $k = 0$ and tends to 0 as $k$ grows. Because it never
exceeds 1, multiplying by it can only shrink a mode and can never cause
divergence, for any $k$ or $dt$. The explicit scheme, by contrast, multiplies a
mode by $(1 - \nu k^2 dt)$, whose magnitude exceeds 1 for large $k \cdot dt$ and
therefore amplifies the mode.

The integrating-factor update handles the stiff linear part by its exact
exponential and steps only the mild nonlinear part explicitly:

$$\hat{u}_{\text{next}} = e^{-\nu k^2 dt} \cdot \left( \hat{u}_{\text{now}} + dt \cdot \hat{N}_{\text{now}} \right)$$

The stiff stability limit is removed, leaving only the mild advective limit from
the nonlinear term, so $dt$ can be much larger.

The nonlinear product $u \cdot u_x$ creates frequencies higher than the grid can
represent, which fold back as false low-frequency noise, an effect called
aliasing, and can corrupt or destabilize the solution. The remedy is the
two-thirds rule: the top third of wavenumbers in the nonlinear term are set to
zero before it is used. This is inexpensive and removes the aliasing.

Together, the integrating factor controls stiffness and the two-thirds dealiasing
controls the nonlinear product, giving a solver that remains stable at large $dt$
and high resolution, so that grid-1024 data generation is fast. The explicit
routine `solve_burgers` is retained for comparison, and the robust routine is
added as `solve_burgers_if`.

---

## 6. Verifying the solver

The solver's output is treated as the true $u(x, 1)$, yet every time-stepper
carries an error that depends on $dt$; for the first-order scheme used here that
error is proportional to $dt$. Because no exact solution is available for
comparison, the check is one of self-consistency: whether the answer stops
changing as $dt$ shrinks. The equation is solved at $dt$, $dt/2$, $dt/4$, and so
on, the finest run is taken as a reference, and the relative L2 error of the
coarser runs against it is measured. If refining $dt$ barely changes the result,
the time error is already negligible and the solver is converged.

The results for the integrating-factor solver at grid 256 and $\nu = 0.01$ are:

| $dt$ | rel-L2 vs finest | change when $dt$ halved |
| --- | --- | --- |
| 2.0e-3 | 2.2e-3 | 1.2e-3 |
| 1.0e-3 | 1.0e-3 | 5.9e-4 |
| 5.0e-4 | 4.4e-4 | 2.9e-4 |
| 2.5e-4 | 1.5e-4 | 1.5e-4 |

At the working step $dt = 10^{-3}$, halving it changes the answer by only about
$6 \times 10^{-4}$, or 0.06 percent, so the criterion is met. This confirms two
things. The solver is converged, since the error is small at the working step
size and the data is therefore trustworthy. The scheme is also of the correct
order, since each halving of $dt$ roughly halves the error, the signature of a
first-order method and a straight line of slope near 1 on a log-log plot; an
erratic or stalled error would instead indicate an implementation bug, so the
clean trend serves as a sanity check.

---

## 7. Downsampling to several resolutions

The defining property of the FNO is resolution invariance: a single trained model
can run on any grid size. The resolution-transfer experiment tests this by
training at a coarse grid, such as 64, and evaluating at finer ones, such as 128,
256, and 1024. For the comparison to be fair, the same solutions must be
represented at each resolution. The data is therefore generated once at a high
base resolution and then downsampled, so that the coarse versions are exact
restrictions of the same fields rather than separately generated ones.

Spectral downsampling is used. Naively keeping every k-th grid point can alias and
distort the field. Instead the field is transformed with the FFT, only the lowest
wavenumbers the coarse grid can represent are kept, up to its Nyquist limit, and
the inverse transform is taken at the smaller size. Because a smooth field's
energy resides in its low modes, discarding the high ones is nearly lossless and
is the natural way to represent a band-limited function on fewer points. A scale
factor of target over source corrects the FFT normalization so that amplitudes
are preserved.

---

## 8. The spectral convolution

The spectral convolution performs the same transform, multiply, invert sequence
as the solver in Concept 1, but the multiplier is learned rather than the fixed
$ik$:

1. Transform the input field to Fourier space.
2. Keep only the lowest `modes` coefficients and discard the rest.
3. Multiply each kept coefficient by a learned complex weight.
4. Transform back to a field.

$$(\mathcal{K}v)(x) = \mathcal{F}^{-1}\big(R \cdot \mathcal{F}(v)\big)(x)$$

where $R$ are the learned per-mode weights, a small complex tensor.

The learned quantity is not a kernel in physical space. The filter is
parametrized directly by its Fourier coefficients $R$, a small set of complex
numbers. The FFT does not discover a kernel; it moves the signal into and out of
the space where the parameters live. By the convolution theorem a multiplication
in Fourier space equals a convolution in physical space, so $R$ implicitly defines
a global convolution filter spanning the whole domain, unlike the small local
kernel of a conventional CNN. This global reach is why the FNO captures
long-range structure inexpensively.

Only the low modes are kept, for two reasons. The first is regularization and
efficiency: the important structure of the solution lives in the low frequencies,
the same intuition as in the Gaussian random field and in downsampling, so
learning weights for only those modes is sufficient and generalizes better. The
second is resolution independence: the lowest 16 modes correspond to the same
physical frequencies, 0 through 15 cycles across the domain, on any grid, and both
a grid of 64 and a grid of 1024 contain them, so a layer that learns 16 complex
weights applies to any resolution unchanged. This is the basis of the
resolution-transfer property.

The contrast with the solver is exact. The solver transforms, multiplies by the
fixed operator $ik$, and inverts, producing a known derivative. The FNO
transforms, multiplies by the learned operator $R$, and inverts, producing a
learned operator. The machinery is the same; only the multiplier differs, fixed
versus trainable.

---

## 9. The full FNO architecture

The spectral convolution of Concept 8 is the core component, wrapped in a standard
lift, process, project structure. The input is $u(x, 0)$ with shape
(batch, grid, in_channels).

1. Lift. A linear layer raises `in_channels` to a wider hidden dimension `width`,
   for example 64, giving the network room to represent features. The input has
   `in_channels = 2`, the field together with its x-coordinate; supplying the grid
   coordinate is a small standard device that helps the model.
2. Fourier layers, repeated for a chosen depth such as 4. Each layer runs two
   paths in parallel and adds them, then applies a GELU nonlinearity. One path is
   the spectral convolution, global but retaining only the low modes. The other is
   a pointwise 1x1 convolution, local and acting at each grid point independently,
   but carrying all frequencies, including the high ones the spectral path
   discards.

   $$x \leftarrow \text{GELU}\big(\text{SpectralConv}(x) + W x\big)$$

3. Project. Two linear layers reduce `width` to `out_channels`, the predicted
   field, with a GELU between them.

The output is the predicted $u(x, 1)$ with shape (batch, grid, out_channels).

Both paths are necessary. The spectral convolution is global but discards high
modes, while the 1x1 convolution is local but retains them; added together they
cover both global low-frequency structure and local high-frequency detail, and
neither alone is sufficient. This matters particularly for shocks, which are sharp
high-frequency features that the spectral path removes and the local path
preserves. The nonlinearity is also necessary: without the GELU, stacking layers
would collapse to a single linear operator, whereas the nonlinearity between
layers allows the model to represent a nonlinear solution operator, which Burgers
requires.

One implementation detail concerns tensor layout. Linear layers expect channels
last, in the order (batch, grid, width), while the convolution-style layers expect
channels in the middle, (batch, width, grid). The forward pass permutes between
the two conventions, which is bookkeeping rather than mathematics.

---

## 10. Training the FNO

Training is standard supervised learning in which the FNO, the student, learns to
reproduce the solver, the teacher. A few choices are specific to operator
learning. The input is $X = [u_0, x\text{-coordinate}]$, two channels, and the
target is $y = u_T$. The loss is the relative L2 error described below, the
optimizer is Adam with weight decay, the learning rate follows a schedule, and
batches are supplied by a data loader.

For each sample the relative L2 loss is
$\lVert \text{pred} - \text{target}\rVert / \lVert \text{target}\rVert$, averaged
over the batch. It is preferred over plain mean squared error because it
normalizes each sample by its own magnitude, so that large- and small-amplitude
fields contribute equally, whereas mean squared error would let large-magnitude
samples dominate. Being scale-invariant, it is comparable across samples,
resolutions, and datasets; it is the standard FNO metric and the same measure used
to verify the solver in Concept 6.

The field channel and the targets are standardized to zero mean and unit variance
using training statistics, so that the optimizer receives well-scaled inputs.
Predictions are decoded back to physical units before the error is measured, so
that the reported figure matches the solver's real output.

At grid 256 with 1000 training samples the test relative L2 error is a fraction of
a percent, so the FNO reproduces the solver closely, whereas random weights would
give an error near 1, or 100 percent. All settings, including grid, modes, width,
depth, epochs, learning rate, and training size, reside in a YAML configuration
file, so that the experiments that follow, namely the baseline comparison,
resolution transfer, and data efficiency, are expressed as different
configurations rather than new code.

---

## 11. Experiment results

All results are at grid 256 unless stated otherwise, measured as relative L2 error
on 200 test samples, over 250 epochs under a matched protocol of Adam with a
learning rate of 1e-3 halved every 100 epochs. The dataset used here is mild and
diffusion-dominated, so its absolute errors are lower than those of the original
paper's Burgers setup and are not directly comparable.

On accuracy, the FNO reaches 0.34 percent error with 287k parameters, against the
U-Net's 1.5 percent with 429k parameters, so the FNO is about 4.4 times more
accurate at comparable size.

Resolution transfer, training at grid 64 and evaluating at 64, 128, and 256, is
the central result. The FNO's error is flat at 0.0029 across every resolution,
while the U-Net degrades sharply, from 0.010 at 64 to 0.37 at 128 and 0.48 at 256.
The difference is structural: the FNO learns weights on low Fourier modes, which
are the same physical frequencies at any grid as described in Concept 8, whereas
the U-Net's convolution kernels are tied to the training grid and encounter
features at the wrong scale when the resolution changes.

On data efficiency, FNO test error against training size runs from 0.73 percent at
100 samples to 0.67 percent at 250, 0.29 percent at 500, and 0.34 percent at 1000,
where the last two differ only by run-to-run noise. Even 100 samples yield
sub-percent error, so the operator is learned from little data.

On speed, for a batch of 200, the numerical solver takes 7.77 ms per sample
against the FNO's 0.33 ms per sample, roughly 23 times faster, because the learned
operator replaces step-by-step simulation with a single forward pass. The
comparison holds hardware, CPU, and batching fixed and excludes the FNO's one-time
training cost.

In summary, the FNO matches or exceeds a strong U-Net on accuracy, is considerably
more data-efficient, runs much faster than the solver, and uniquely transfers
across resolutions. The resolution-transfer result is the clean, structural
advantage.

---

## 12. Methodology note: making the baseline fair

The U-Net comparison was not accepted at face value. An initial bare convolutional
U-Net underfit badly, at about 28 percent training error, which suggested that a
U-Net could not perform the task. That conclusion was treated as too convenient
and investigated.

An overfit test on a tiny batch showed that the U-Net could fit 16 samples,
reaching about 4 percent, which established that the code was sound and that there
was no fundamental representational barrier. A sweep over depth showed that
performance improved only once the receptive field became near-global, at five
levels, confirming that the bottleneck was global information flow, though at a
large parameter cost. The decisive change was the addition of GroupNorm and
residual connections, which reduced the three-level U-Net from about 28 percent to
about 2.8 percent, showing that the original failure was an under-engineered
baseline rather than a fundamental limit.

Only after these steps was the comparison run. The lesson is that a negative
result is only as strong as the effort invested in the losing side, and a baseline
must be genuinely good before its loss carries any meaning.

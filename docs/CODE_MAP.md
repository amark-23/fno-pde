# Code map

A short guide to the repository, one entry per file. Function docstrings hold the
detail; this is for navigation.

## src/solvers/burgers.py
The 1D spectral Burgers solver and dataset generation.
- `gaussian_random_field(...)` samples smooth random initial conditions u(x, 0).
- `solve_burgers(...)` is the simple explicit (Euler) time-stepping solver.
- `solve_burgers_if(...)` is the robust integrating-factor solver with two-thirds
  dealiasing; it generates the ground-truth data.
- `downsample(u, target_grid)` spectrally downsamples a field to a coarser grid.
- `generate_dataset(...)` builds the (u0, uT) pairs at several resolutions.
- `main()` is the CLI entry point that writes `data/burgers.npz`.

## src/models/fno.py
The Fourier Neural Operator, from scratch.
- `SpectralConv1d` is the learned Fourier-space multiply on the lowest modes.
- `FNO1d` wraps it: lift, stacked Fourier layers (spectral path plus a 1x1
  convolution), then project.

## src/models/unet.py
The convolutional baseline.
- `ResBlock` is a Conv, GroupNorm, GELU block with a residual skip.
- `UNet1d` is a configurable-depth encoder and decoder built from those blocks.
- `_groups(c)` picks a valid GroupNorm group count.

## src/data.py
Data loading and normalization.
- `Normalizer` standardizes with training statistics and inverts the transform.
- `load_split(...)` loads a train or test split and attaches the x-coordinate
  channel.
- `make_loader(...)` wraps tensors in a DataLoader.

## src/metrics.py
- `relative_l2(pred, target)` is the mean relative L2 error.
- `RelativeL2Loss` exposes the same computation as a loss module.

## src/train.py
Config-driven training.
- `train_one(cfg, device)` builds, trains, and evaluates one model, returning the
  model and a result row.
- `append_result(row, path)` appends a result to `results/metrics.csv`.
- `main()` reads a YAML config and runs a single training job.

## configs/
- `burgers_fno.yaml` is the FNO training config.
- `burgers_unet.yaml` is the U-Net baseline config.

## experiments/
- `_common.py` holds shared helpers: `build_model`, `train_model`, `eval_at`,
  `n_params`.
- `run_accuracy.py` is Experiment 1, FNO against U-Net at grid 256.
- `run_resolution_transfer.py` is Experiment 2, train at 64 and evaluate at
  64, 128, 256.
- `run_data_efficiency.py` is Experiment 3, error against training-set size.
- `run_speed.py` is Experiment 4, FNO inference against the numerical solver.
- `figures/` holds the output figures used in the README.

## Part 2: 2D Navier-Stokes

### src/solvers/navier_stokes.py
2D Navier-Stokes solver, NumPy reference, vorticity-streamfunction form.
- `spectral_grid(N)` builds the 2D wavenumbers and dealias mask.
- `vorticity_to_velocity(...)` does the Poisson solve and streamfunction step.
- `rhs(...)` forms the advection term plus forcing in Fourier space.
- `solve_ns(...)` steps forward with the integrating factor, saving snapshots.
- `gaussian_random_field_2d(...)` and `generate_dataset(...)` build trajectories.

### src/solvers/navier_stokes_torch.py
A faithful PyTorch/GPU port of the above, used for dataset generation at scale.

### src/data_ns.py
2D data loading: `single_step_pairs`, `add_coords`, `Normalizer`, `load_ns`,
`make_loader`.

### src/experiments_ns.py
`downsample_2d` (spectral downsampling) and `rollout` (autoregressive prediction)
for the 2D experiments.

### 2D additions to existing files
- `src/models/fno.py` also defines `SpectralConv2d` and `FNO2d`.
- `src/models/unet.py` also defines `ResBlock2d` and `UNet2d`.
- `notebooks/` holds the Colab notebooks for GPU generation and training.


## src/models/fno_nd.py (dimension-general)
The spectral convolution and FNO written for any number of spatial dimensions.
- `SpectralConvNd` generalizes `SpectralConv1d` and `SpectralConv2d`: it uses
  2**(d-1) corner blocks and an einsum built from d spatial labels, and is
  bit-exact to the fixed-dimension layers when given the same weights.
- `FNONd` wraps it with a channel-wise linear local path in place of a
  fixed-dimension convolution, so one class runs in 1D, 2D, or 3D.

## tests/
- `test_spectral_nd.py` checks `SpectralConvNd` against `SpectralConv1d` and
  `SpectralConv2d`, and runs `FNONd` forward and backward in 1D, 2D, and 3D.

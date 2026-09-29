# Development notes

Selected figures from the exploratory work behind the project, in the order the
work happened. They record how the solver was built and verified and how the FNO
was validated. The working scripts remain in `sandbox/`.

## 1. Initial conditions (Gaussian random field)
![GRF samples](figures/grf_samples.png)

Four smooth random curves drawn from the same Gaussian random field. These become
the u(x, 0) inputs.

## 2. Shock formation
![Shock](figures/shock.png)

A smooth field steepening into a shock and then diffusing as Burgers evolves.

## 3. Space-time view
![Space-time](figures/space_time.png)

The full field u(x, t). A snapshot is one horizontal slice, and the diagonal
front is the shock moving through space over time.

## 4. Solver stability
![Stability](figures/stability_growth.png)

At dt = 1e-3 the explicit scheme diverges to NaN, while the integrating-factor
solver stays bounded. This motivated the robust stepper.

## 5. Convergence check
![Convergence](figures/convergence.png)

Relative error against dt, showing first-order convergence and confirming the
solver is trustworthy as ground truth.

## 6. Spectral downsampling
![Downsample](figures/downsample.png)

Coarse copies at 128 and 64 points land exactly on the fine 256-point curve.

## 7. Training data
![Example pairs](figures/example_pairs.png)

Five (input u0, target uT) pairs from the generated dataset.

## 8. FNO prediction
![Prediction](figures/prediction_vs_truth.png)

The trained FNO's prediction lies on top of the solver's ground truth.

# burgers_solver — sandbox tests

Throwaway experiments while building and verifying the Burgers' solver.

- `scripts/` — runnable test scripts (path-robust; run from anywhere).
- `figures/` — plots produced while exploring:
  - `shock.png` — a smooth curve steepening into a shock across several times.
  - `space_time.png` — snapshot view vs the full u(x, t) as a space-time heatmap.
  - `sharpness_vs_time.png` — shock sharpness (max slope) rises, peaks, then decays.
  - `different_ics.png` — different initial conditions produce different shocks.
  - `stability_growth.png` — explicit (dt=1e-3) blows up; the IF solver stays stable.
  - `convergence.png` — rel-L2 error vs dt: first-order convergence (verification).

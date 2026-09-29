# Sandbox experiment log

Terse record of investigations run in `sandbox/`. Throwaway; deleted before publish.

## 2026-09-28 — Solver (data factory)
- GRF initial conditions: verified smooth, zero-mean; alpha = smoothness knob.
- Explicit Burgers' solver: shock forms then diffuses; max-slope peaks ~t=0.29.
- Stability: explicit blows up (NaN) at dt=1e-3; step-size limit is real.
- Integrating-factor (robust) solver: stable at 10x dt, matches explicit to ~1e-3.
- Convergence check: first-order in dt; error halves as dt halves -> trustworthy.
- Downsample (spectral): coarse copies land exactly on fine curve.
- Dataset: 1000+200 pairs, res {64,128,256}, generated in ~15 s.

## 2026-09-29 — Models & experiments
- SpectralConv1d: runs at 64 & 256 with same weights; gradients flow. OK.
- FNO1d: full model, 287k params, resolution-independent, gradients OK.
- FNO training: test rel-L2 ~0.003 (our data easier than paper's ~1.4%).
- U-Net baseline saga (accuracy, Exp 1):
  - Bare conv U-Net (2- & 3-level): stuck ~20-28% TRAIN loss. Suspicious.
  - Ruled out LR (still stuck at 1e-4) and reporting noise (epoch-avg still flat).
  - Overfit-16-samples test: U-Net reaches 4% -> NOT structural, code is fine.
  - Depth sweep (bare): plateau breaks only at 5 levels/6.1M params (~18%);
    receptive field is the lever, but efficiency gap is huge.
  - KEY FIX: added GroupNorm + residual blocks -> 3-level drops 28% -> 2.8%.
    The earlier "U-Net can't fit" was a strawman (under-engineered), not a limit.
- Clean Exp 1 result (250 ep, matched protocol):
  - FNO 287k -> test 0.0035 ; U-Net 429k -> test 0.0152. FNO ~4.4x better.

## Lesson
A negative result is only as strong as the effort on the losing side.
Make the baseline genuinely good before concluding it lost.

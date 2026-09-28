# FNO for PDEs

Learning PDE solution operators with **Fourier Neural Operators (FNOs)**, benchmarked against a U-Net and a classical numerical solver.

This repository accompanies a master's-thesis-adjacent portfolio project. The research question:

> How do FNOs compare to a standard CNN baseline and a numerical solver on **accuracy**, **speed**, **data efficiency** and **generalization to unseen grid resolutions**?

Two PDEs are covered: the 1D viscous Burgers' equation (data generated here from a spectral solver) and 2D incompressible Navier–Stokes in vorticity form (external dataset).

> **Status:** scaffold with working starter code. Results and figures are filled in as the experiments are run — see the [project plan](https://claude.ai/code/artifact/a9639750-549a-4871-90c2-9ec5c3094042).

## Method in one paragraph

A Fourier Neural Operator learns a map between *functions* rather than between fixed-size vectors. Its core layer is a **spectral convolution**: transform the input to Fourier space, keep the lowest few modes, multiply them by learned complex weights, and transform back. Because the weights act on frequencies rather than grid points, a model trained on a coarse grid can be evaluated on a finer one — the property this project tests directly.

The retained-mode spectral update, per Fourier mode k below a cutoff:

```latex
(\mathcal{K}v)(x) = \mathcal{F}^{-1}\!\big(R \cdot \mathcal{F}(v)\big)(x)
```

where F is the (discrete) Fourier transform and R are the learned per-mode weights.

## Repo layout

```
fno-pde/
├── README.md
├── requirements.txt
├── src/
│   ├── solvers/burgers.py     # spectral Burgers' solver + dataset generation
│   ├── models/fno.py          # FNO1d, FNO2d (from scratch)
│   ├── models/unet.py         # UNet1d, UNet2d baselines
│   ├── data.py                # loaders, normalization
│   ├── train.py               # config-driven training loop
│   └── metrics.py             # relative L2 error, timing, param counts
├── configs/                   # one YAML per experiment
├── notebooks/                 # 01_burgers, 02_navier_stokes
└── results/                   # metrics.csv, figures, GIFs
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the Burgers' dataset (a few minutes on CPU)
python -m src.solvers.burgers --n_samples 1200 --grid 1024 --out data/burgers.npz

# 2. Train the FNO and the U-Net baseline
python -m src.train --config configs/burgers_fno.yaml
python -m src.train --config configs/burgers_unet.yaml
```

Each run appends a row (test error, parameter count, wall-clock) to `results/metrics.csv`.

On Colab, open `notebooks/01_burgers.ipynb` and run top to bottom; a free T4 is plenty for the 1D experiments.

## Experiments

| # | Experiment | What it shows |
|---|---|---|
| 1 | Accuracy | Relative L2 error, FNO vs U-Net (3 seeds) |
| 2 | Resolution transfer | Train at 64 points, test at 128 / 256 / 1024 — the FNO generalizes, the U-Net can't directly |
| 3 | Data efficiency | Error vs training-set size (100 → 1000) |
| 4 | Long rollouts (2D) | Autoregressive error growth over 20+ steps |
| 5 | Speed | Inference time vs the numerical solver |

## Results

_Filled in during Week 2–3. Headline table and the true-vs-predicted vorticity animation go here._

## References

- Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021 — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces*, JMLR 2023 — [arXiv:2108.08481](https://arxiv.org/abs/2108.08481)
- Takamoto et al., *PDEBench*, NeurIPS 2022 — [arXiv:2210.07182](https://arxiv.org/abs/2210.07182)
- Reference implementation: [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator)

## License

MIT — see [LICENSE](LICENSE).

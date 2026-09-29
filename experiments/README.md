# experiments

Scripts that run the FNO/U-Net experiments from `src/` and produce the tables and
figures for the README. Each reads configs from `configs/` and writes results to
`results/`.

Planned:
- `run_accuracy.py`          - FNO vs U-Net test error at grid 256 (Experiment 1)
- `run_resolution_transfer.py` - train at 64, evaluate at 64/128/256 (Experiment 2)
- `run_data_efficiency.py`   - error vs training-set size (Experiment 3)
- `run_speed.py`             - FNO inference vs the numerical solver (Experiment 5)

Run from the repo root, e.g.:  `py experiments/run_accuracy.py`

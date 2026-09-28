"""Config-driven training loop for the 1D Burgers' experiments.

Example:
    python -m src.train --config configs/burgers_fno.yaml

Each YAML config specifies the model, data resolution, training-set size and
optimizer settings. Results (final test error, params, wall-clock) are appended
to results/metrics.csv so the plots and tables regenerate from files.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch
import yaml

from .data import load_burgers, make_loaders, GaussianNormalizer
from .metrics import RelativeL2Loss, relative_l2, count_params
from .models.fno import FNO1d
from .models.unet import UNet1d


def build_model(cfg):
    name = cfg["model"].lower()
    if name == "fno":
        return FNO1d(modes=cfg.get("modes", 16), width=cfg.get("width", 64),
                     depth=cfg.get("depth", 4))
    if name == "unet":
        return UNet1d(width=cfg.get("width", 32))
    raise ValueError(f"unknown model {name!r}")


def train_one(cfg, device):
    torch.manual_seed(cfg.get("seed", 0))
    (x_tr, y_tr), (x_te, y_te) = load_burgers(
        cfg["data"], grid=cfg.get("grid"),
        n_train=cfg.get("n_train", 1000), n_test=cfg.get("n_test", 200))

    # Normalize inputs/outputs with training statistics.
    x_norm = GaussianNormalizer(x_tr[..., 0])
    y_norm = GaussianNormalizer(y_tr)
    x_tr = x_tr.clone(); x_te = x_te.clone()
    x_tr[..., 0] = x_norm.encode(x_tr[..., 0]); x_te[..., 0] = x_norm.encode(x_te[..., 0])
    y_tr_n = y_norm.encode(y_tr)

    tr, _ = make_loaders(x_tr, y_tr_n, x_te, y_te, batch_size=cfg.get("batch_size", 32))

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 100)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = RelativeL2Loss()

    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"epoch {epoch + 1}/{epochs}  train rel-L2 {loss.item():.4f}")
    train_time = time.perf_counter() - start

    # Evaluate in physical units (decode the prediction).
    model.eval()
    with torch.no_grad():
        pred = y_norm.decode(model(x_te.to(device)).cpu())
        test_err = relative_l2(pred, y_te).item()

    return {"model": cfg["model"], "grid": cfg.get("grid") or "full",
            "n_train": cfg.get("n_train", 1000), "seed": cfg.get("seed", 0),
            "test_rel_l2": round(test_err, 5), "params": count_params(model),
            "train_time_s": round(train_time, 1), "epochs": epochs}


def append_result(row, path="results/metrics.csv"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  config: {args.config}")
    row = train_one(cfg, device)
    print("Result:", row)
    append_result(row)


if __name__ == "__main__":
    main()

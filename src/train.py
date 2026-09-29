"""Config-driven training for the FNO on Burgers'.

Run:  py -m src.train --config configs/burgers_fno.yaml
"""
from .models.fno import FNO1d
from .models.unet import UNet1d
import argparse
import csv
import os
import time

import torch
import yaml

from .data import load_split, make_loader, Normalizer
from .metrics import RelativeL2Loss, relative_l2



def train_one(cfg, device):
    torch.manual_seed(cfg.get("seed", 0))
    npz = cfg["data"]
    grid = cfg["grid"]

    # 1. Load train/test at the configured resolution.
    x_tr, y_tr = load_split(npz, grid, "train", n=cfg.get("n_train"))
    x_te, y_te = load_split(npz, grid, "test",  n=cfg.get("n_test"))

    # 2. Normalize the field channel (and targets) using TRAIN statistics.
    x_norm = Normalizer(x_tr[..., 0])
    y_norm = Normalizer(y_tr)
    x_tr = x_tr.clone(); x_te = x_te.clone()
    x_tr[..., 0] = x_norm.encode(x_tr[..., 0])
    x_te[..., 0] = x_norm.encode(x_te[..., 0])
    y_tr_n = y_norm.encode(y_tr)

    train_loader = make_loader(x_tr, y_tr_n, batch_size=cfg.get("batch_size", 32))

    # 3. Model, optimizer, schedule, loss.
    if cfg.get("model", "fno") == "unet":
        model = UNet1d(width=cfg.get("width", 24),
                       n_levels=cfg.get("n_levels", 3)).to(device)
    else:
        model = FNO1d(modes=cfg.get("modes", 16), width=cfg.get("width", 64),
                      depth=cfg.get("depth", 4)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 100)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("lr_step", 100),
                                            gamma=cfg.get("lr_gamma", 0.5))
    loss_fn = RelativeL2Loss()

    # 4. Training loop.
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item(); n_batches += 1
        sched.step()
        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  train rel-L2 {epoch_loss / n_batches:.4f}")
    train_time = time.perf_counter() - start

    # 5. Evaluate on test, in physical units (decode the prediction).
    model.eval()
    with torch.no_grad():
        pred = y_norm.decode(model(x_te.to(device)).cpu())
        test_err = relative_l2(pred, y_te).item()

    return model, {
        "model": cfg.get("model", "fno"), "grid": grid, "n_train": cfg.get("n_train", len(x_tr)),
        "seed": cfg.get("seed", 0), "test_rel_l2": round(test_err, 5),
        "params": sum(p.numel() for p in model.parameters()),
        "train_time_s": round(train_time, 1), "epochs": epochs,
    }


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
    _, row = train_one(cfg, device)
    print("Result:", row)
    append_result(row)


if __name__ == "__main__":
    main()
"""
Config-driven rollout-stabilized training for the 2D Navier-Stokes FNO.

The single-step task trains omega(t) -> omega(t+1) on ground-truth inputs, which
leaves the model exposed to its own errors only at inference (Concept 17). This
module adds two training schemes that reduce that drift, both selected by config:

- Unrolled / pushforward training (rollout_steps > 1): the model is applied for
  several steps during training, feeding each prediction back in, so it learns on
  its own outputs. With pushforward true, only the final step carries a gradient
  and the earlier steps run detached, which supplies a realistic off-distribution
  input without backpropagating through the whole rollout. With pushforward false,
  the loss is averaged over all steps and the gradient flows through the rollout.
- Noise injection (noise_std > 0): small Gaussian noise is added to the input each
  step, so the model learns to correct slightly perturbed fields.

rollout_steps = 1 with noise_std = 0 recovers the standard single-step baseline.

Run:  python -m src.train_ns --config configs/ns_pushforward.yaml
"""
from .models.fno import FNO2d
from .models.unet import UNet2d
import argparse
import time

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from .data_ns import Normalizer, single_step_pairs
from .metrics import relative_l2


def load_ns_traj(path, n_train_traj, n_test_traj):
    """Load raw vorticity trajectories and the train-set normalizers."""
    vort = torch.from_numpy(np.load(path)["vorticity"]).float()   # (n_traj, T, N, N)
    train = vort[:n_train_traj]
    test = vort[n_train_traj:n_train_traj + n_test_traj]
    x_tr, y_tr = single_step_pairs(train)                          # raw omega(t), omega(t+1)
    return train, test, Normalizer(x_tr), Normalizer(y_tr)


def _add_coords(field):
    """(B, N, N, 1) -> (B, N, N, 3): append x and y coords on the field's device."""
    B, N = field.shape[0], field.shape[1]
    dev = field.device
    gx = torch.linspace(0, 1, N, device=dev).view(1, N, 1, 1).expand(B, N, N, 1)
    gy = torch.linspace(0, 1, N, device=dev).view(1, 1, N, 1).expand(B, N, N, 1)
    return torch.cat([field, gx, gy], dim=-1)


def _norm_to(norm, device):
    norm.mean = norm.mean.to(device)
    norm.std = norm.std.to(device)
    return norm


class WindowDataset(Dataset):
    """Consecutive windows of length rollout_steps + 1 from raw trajectories."""

    def __init__(self, traj, k):
        self.traj = traj
        self.k = k
        self.per = traj.shape[1] - k
        assert self.per > 0, "trajectory shorter than the rollout window"

    def __len__(self):
        return self.traj.shape[0] * self.per

    def __getitem__(self, idx):
        i, t = divmod(idx, self.per)
        return self.traj[i, t:t + self.k + 1]                      # (k+1, N, N)


def rollout_loss(model, window, x_norm, y_norm, k, pushforward, noise_std):
    """Relative-L2 loss over a k-step rollout starting from window[:, 0].

    window: (B, k+1, N, N) raw vorticity. Returns a scalar loss.
    """
    def step(w):
        xf = x_norm.encode(w.unsqueeze(-1))                        # (B, N, N, 1)
        if noise_std > 0:
            xf = xf + noise_std * torch.randn_like(xf)
        return model(_add_coords(xf))                              # (B, N, N, 1) normalized

    w = window[:, 0]
    if pushforward and k > 1:
        with torch.no_grad():                                     # off-distribution input, no BPTT
            for s in range(1, k):
                w = y_norm.decode(step(w))[..., 0]
        pred = step(w)                                            # only the final step carries a gradient
        target = y_norm.encode(window[:, k].unsqueeze(-1))
        return relative_l2(pred, target)

    total = 0.0                                                    # full unrolled (k >= 1)
    for s in range(1, k + 1):
        pred = step(w)
        target = y_norm.encode(window[:, s].unsqueeze(-1))
        total = total + relative_l2(pred, target)
        w = y_norm.decode(pred)[..., 0]
    return total / k


@torch.no_grad()
def eval_rollout_error(model, test_traj, x_norm, y_norm, n_steps, device):
    """Relative-L2 per rollout step on the test trajectories (physical units)."""
    model.eval()
    w = test_traj[:, 0].to(device)                                # (B, N, N) raw
    true = test_traj[:, 1:1 + n_steps].to(device)                 # (B, n_steps, N, N) raw
    errs = []
    for s in range(n_steps):
        xf = x_norm.encode(w.unsqueeze(-1))
        w = y_norm.decode(model(_add_coords(xf)))[..., 0]         # raw predicted next
        num = (w - true[:, s]).reshape(w.shape[0], -1).norm(dim=1)
        den = true[:, s].reshape(w.shape[0], -1).norm(dim=1).clamp_min(1e-8)
        errs.append((num / den).mean().item())
    return errs


def build_model(cfg):
    if cfg["model"] == "fno":
        return FNO2d(modes1=cfg.get("modes1", 12), modes2=cfg.get("modes2", 12),
                     width=cfg.get("width", 32), depth=cfg.get("depth", 4),
                     in_channels=3, out_channels=1)
    if cfg["model"] == "unet":
        return UNet2d(in_channels=3, out_channels=1,
                      width=cfg.get("width", 24), n_levels=cfg.get("n_levels", 3))
    raise ValueError(f"unknown model {cfg['model']!r}")


def train_rollout(cfg, device):
    torch.manual_seed(cfg.get("seed", 0))
    k = cfg.get("rollout_steps", 1)
    pushforward = cfg.get("pushforward", False)
    noise_std = cfg.get("noise_std", 0.0)

    train_traj, test_traj, x_norm, y_norm = load_ns_traj(
        cfg["data"], cfg.get("n_train_traj", 800), cfg.get("n_test_traj", 200))
    x_norm, y_norm = _norm_to(x_norm, device), _norm_to(y_norm, device)

    model = build_model(cfg).to(device)
    loader = DataLoader(WindowDataset(train_traj, k),
                        batch_size=cfg.get("batch_size", 32), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 50)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.5)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = nb = 0
        for window in loader:
            window = window.to(device)
            opt.zero_grad()
            loss = rollout_loss(model, window, x_norm, y_norm, k, pushforward, noise_std)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if (ep + 1) % max(1, epochs // 10) == 0:
            print(f"  epoch {ep+1}/{epochs}  train loss {tot/nb:.4f}")

    curve = eval_rollout_error(model, test_traj, x_norm, y_norm,
                               cfg.get("eval_rollout_steps", 19), device)
    print(f"  single-step test rel-L2 {curve[0]:.4f} | "
          f"rollout end {curve[-1]:.4f} | {time.time()-t0:.0f}s")
    return model, {"single_step": curve[0], "rollout_curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{cfg.get('model')} | rollout_steps {cfg.get('rollout_steps', 1)} | "
          f"pushforward {cfg.get('pushforward', False)} | noise_std {cfg.get('noise_std', 0.0)}")
    train_rollout(cfg, device)


if __name__ == "__main__":
    main()

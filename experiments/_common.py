"""Shared helpers for the experiment scripts."""
import os, sys, time
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
NPZ = os.path.join(REPO, "data", "burgers.npz")
FIGDIR = os.path.join(REPO, "experiments", "figures")
os.makedirs(FIGDIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.data import load_split, make_loader, Normalizer
from src.metrics import relative_l2
from src.models.fno import FNO1d
from src.models.unet import UNet1d


def build_model(cfg):
    if cfg.get("model", "fno") == "unet":
        return UNet1d(width=cfg.get("width", 24), n_levels=cfg.get("n_levels", 3))
    return FNO1d(modes=cfg.get("modes", 16), width=cfg.get("width", 64),
                 depth=cfg.get("depth", 4))


def train_model(cfg, train_grid, epochs=250):
    """Train a model at train_grid; return (model, x_norm, y_norm)."""
    torch.manual_seed(cfg.get("seed", 0))
    x_tr, y_tr = load_split(NPZ, train_grid, "train", n=cfg.get("n_train"))
    xn, yn = Normalizer(x_tr[..., 0]), Normalizer(y_tr)
    x_tr = x_tr.clone(); x_tr[..., 0] = xn.encode(x_tr[..., 0])
    loader = make_loader(x_tr, yn.encode(y_tr), batch_size=cfg.get("batch_size", 32))

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.get("lr_step", 100),
                                            gamma=cfg.get("lr_gamma", 0.5))
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); relative_l2(model(xb), yb).backward(); opt.step()
        sched.step()
    return model, xn, yn


def eval_at(model, xn, yn, grid, n=None):
    """Test rel-L2 at a given grid (physical units)."""
    x, y = load_split(NPZ, grid, "test", n=n)
    x = x.clone(); x[..., 0] = xn.encode(x[..., 0])
    model.eval()
    with torch.no_grad():
        pred = yn.decode(model(x.to(device)).cpu())
    return relative_l2(pred, y).item()


def n_params(model):
    return sum(p.numel() for p in model.parameters())

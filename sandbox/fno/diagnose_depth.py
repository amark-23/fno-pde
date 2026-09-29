"""Does more receptive field let the U-Net fit? Run:  py sandbox/fno/diagnose_depth.py

Trains U-Nets of increasing depth on the FULL training set and reports final
TRAIN rel-L2. If deeper (bigger receptive field) => lower train loss, the earlier
failure was a receptive-field limit, not a fundamental one.
"""
import os, sys, time
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from src.data import load_split, make_loader, Normalizer
from src.metrics import relative_l2
from src.models.unet import UNet1d

device = torch.device("cpu")
EPOCHS = 80
GRID = 256

x_tr, y_tr = load_split(os.path.join(REPO, "data", "burgers.npz"), GRID, "train")
x_te, y_te = load_split(os.path.join(REPO, "data", "burgers.npz"), GRID, "test")
xn, yn = Normalizer(x_tr[..., 0]), Normalizer(y_tr)
x_tr = x_tr.clone(); x_te = x_te.clone()
x_tr[..., 0] = xn.encode(x_tr[..., 0]); x_te[..., 0] = xn.encode(x_te[..., 0])
y_tr_n = yn.encode(y_tr)
loader = make_loader(x_tr, y_tr_n, batch_size=32)

def run(n_levels):
    torch.manual_seed(0)
    model = UNet1d(width=24, n_levels=n_levels).to(device)
    params = sum(p.numel() for p in model.parameters())
    coarsest = GRID // (2 ** n_levels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad(); loss = relative_l2(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
    model.eval()
    with torch.no_grad():
        test = relative_l2(yn.decode(model(x_te)), y_te).item()
    return params, coarsest, tot / nb, test, time.perf_counter() - t0

print(f"{'levels':>6} {'params':>10} {'coarsest_grid':>14} {'train_relL2':>12} {'test_relL2':>11} {'sec':>6}")
for n in (3, 4, 5):
    p, c, tr, te, s = run(n)
    print(f"{n:>6} {p:>10,} {c:>14} {tr:>12.4f} {te:>11.4f} {s:>6.0f}")
print("\nIf train_relL2 drops as levels increase -> receptive field was the limit.")

"""Helpers for the 2D Navier-Stokes experiments (resolution transfer, rollout)."""
import torch
from .data_ns import add_coords


def downsample_2d(field, target_N):
    """Spectrally downsample a real field (..., N, N) -> (..., target_N, target_N).

    Keeps the low modes in both axes (two corners on the first axis, one block on
    the last, like the 2D spectral convolution) and rescales the amplitude.
    """
    N = field.shape[-1]
    if target_N >= N:
        return field
    m = target_N // 2
    keep = target_N // 2 + 1
    f_ft = torch.fft.rfft2(field)                                  # (..., N, N//2+1)
    out = torch.zeros(*field.shape[:-2], target_N, keep,
                      dtype=f_ft.dtype, device=field.device)
    out[..., :m, :keep] = f_ft[..., :m, :keep]                     # low +kx
    out[..., -m:, :keep] = f_ft[..., N - m:, :keep]                # low -kx
    return torch.fft.irfft2(out, s=(target_N, target_N)) * (target_N / N) ** 2


def rollout(model, w0, n_steps, x_norm, y_norm, device="cpu"):
    """Autoregressive rollout from initial vorticity w0 (B, N, N).

    Applies the single-step model repeatedly, feeding each prediction back in.
    Returns the predicted trajectory (B, n_steps, N, N).
    """
    model.eval()
    w = w0.clone()
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = add_coords(x_norm.encode(w.unsqueeze(-1)))         # (B, N, N, 3)
            pred = y_norm.decode(model(x.to(device)).cpu())[..., 0]  # (B, N, N)
            preds.append(pred)
            w = pred
    return torch.stack(preds, dim=1)                               # (B, n_steps, N, N)


def animate_vorticity(true_traj, pred_traj, path, fps=8,
                      title="rollout: truth vs FNO"):
    """Animate one vorticity rollout as a 3-panel GIF: truth, prediction, |error|.

    true_traj, pred_traj: arrays or tensors of shape (T, N, N) for a single
    trajectory (the same T steps in each). Truth and prediction share one
    diverging color scale so they are directly comparable; the error panel uses
    its own sequential scale. Writes an animated GIF to `path` and returns it.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    true = np.asarray(true_traj, dtype=float)
    pred = np.asarray(pred_traj, dtype=float)
    T = true.shape[0]
    err = np.abs(pred - true)
    vmax = float(np.abs(true).max())

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3))
    panels = [
        ("truth",   true, dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)),
        ("FNO",     pred, dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax)),
        ("|error|", err,  dict(cmap="magma",  vmin=0.0,   vmax=vmax)),
    ]
    ims = []
    for ax, (name, data, kw) in zip(axes, panels):
        im = ax.imshow(data[0], origin="lower", **kw)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
        ims.append(im)
    sup = fig.suptitle(f"{title}    step 1/{T}")
    fig.tight_layout()

    def update(t):
        for im, (_, data, _) in zip(ims, panels):
            im.set_data(data[t])
        sup.set_text(f"{title}    step {t+1}/{T}")
        return ims

    anim = FuncAnimation(fig, update, frames=T, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path

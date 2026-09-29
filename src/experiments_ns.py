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

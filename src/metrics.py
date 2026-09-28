"""Evaluation metrics and timing helpers."""
from __future__ import annotations

import time
import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean relative L2 error over a batch.

    Both tensors are flattened per sample; returns a scalar tensor. This is the
    standard metric in the FNO literature and is scale-invariant, so it is
    comparable across resolutions.
    """
    b = pred.shape[0]
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)
    num = torch.linalg.norm(pred - target, dim=1)
    den = torch.linalg.norm(target, dim=1).clamp_min(1e-8)
    return (num / den).mean()


class RelativeL2Loss(torch.nn.Module):
    """Relative L2 as a training loss (works better than MSE for these operators)."""

    def forward(self, pred, target):
        return relative_l2(pred, target)


@torch.no_grad()
def time_inference(model, sample, device, n_repeat: int = 50, warmup: int = 5) -> float:
    """Return mean inference time per call (seconds) for a fixed input batch."""
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_repeat):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_repeat


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

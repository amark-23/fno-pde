"""Evaluation metric / training loss for operator learning."""
import torch


def relative_l2(pred, target):
    """Mean relative L2 error over a batch.

    Per sample: ||pred - target|| / ||target||, then averaged. Scale-invariant,
    so it's comparable across samples and resolutions -- the standard FNO metric.
    """
    b = pred.shape[0]
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)
    num = torch.linalg.norm(pred - target, dim=1)
    den = torch.linalg.norm(target, dim=1).clamp_min(1e-8)
    return (num / den).mean()


class RelativeL2Loss(torch.nn.Module):
    """relative_l2 as a loss module (usable like nn.MSELoss)."""
    def forward(self, pred, target):
        return relative_l2(pred, target)
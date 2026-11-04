"""Unreduced IoU-weighted VarifocalNet loss for cached final class scores."""
import torch


def varifocal_elements(logits, target, negative_alpha=.75, gamma=2.,
                      iou_weighted=True, bce=None):
    """Return VFL entries before masking or reducing.

    As in the official implementation, the negative focal weight remains in
    the autograd graph. IoU-weighted positives receive weight q; zero-target
    entries receive alpha * sigmoid(logit)**gamma. The optional raw BCE tensor
    avoids recomputing BCE while retaining the original diagnostics.
    """
    if logits.shape != target.shape:
        raise ValueError('VFL logits and targets must have identical shapes')
    if negative_alpha < 0 or gamma < 0:
        raise ValueError('VFL alpha and gamma must be nonnegative')
    target = target.to(dtype=logits.dtype)
    if bce is None:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction='none')
    elif bce.shape != logits.shape:
        raise ValueError('Unreduced BCE must have the logits shape')
    positive_weight = target if iou_weighted else torch.ones_like(target)
    focal_weight = torch.where(
        target > 0, positive_weight,
        negative_alpha * logits.sigmoid().pow(gamma))
    return bce * focal_weight

"""Bounding-box masks shared by training alignment and evaluation diagnostics."""
from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F


def region_masks(valid_mask: Tensor, targets: list[dict], hw: tuple[int, int],
                 image_hw: tuple[int, int], *, return_counts: bool = False):
    """Select feature-cell centers inside GT boxes, excluding image padding.

    Overlapping boxes count a cell once. Coverage counts boxes containing at
    least one valid feature-cell center; a box may be absent at coarse levels.
    """
    h, w = hw
    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1 or len(targets) != valid_mask.shape[0]:
        raise ValueError('Expected B1HW valid_mask and one target dictionary per image')
    if min(h, w, *image_hw) <= 0:
        raise ValueError('Image and feature sizes must be positive')
    valid = F.interpolate(valid_mask.float(), size=hw, mode='nearest-exact')[:, 0].bool()
    yy, xx = torch.meshgrid(
        (torch.arange(h, device=valid.device) + .5) * image_hw[0] / h,
        (torch.arange(w, device=valid.device) + .5) * image_hw[1] / w, indexing='ij')
    fg = torch.zeros_like(valid)
    covered = torch.zeros((), device=valid.device, dtype=torch.long)
    total = 0
    for index, target in enumerate(targets):
        boxes = target['boxes'].detach().to(device=valid.device, dtype=torch.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError('Expected Nx4 xyxy boxes in letterboxed image coordinates')
        total += len(boxes)
        if len(boxes):
            inside = ((xx[None] >= boxes[:, 0, None, None]) &
                      (xx[None] < boxes[:, 2, None, None]) &
                      (yy[None] >= boxes[:, 1, None, None]) &
                      (yy[None] < boxes[:, 3, None, None]))
            inside &= valid[index][None]
            fg[index] = inside.any(0)
            if return_counts:
                covered += inside.flatten(1).any(1).sum()
    masks = {'all': valid, 'foreground': valid & fg, 'background': valid & ~fg}
    if return_counts:
        return masks, {'objects': covered.new_tensor(total), 'covered_objects': covered}
    return masks

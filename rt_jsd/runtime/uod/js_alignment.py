"""Paired IFE alignment with Jensen-Shannon divergence, without NFE."""
from __future__ import annotations

import math
from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .feature_regions import region_masks


def channel_js(left: Tensor, right: Tensor, temperature: float = 1.0) -> Tensor:
    """JSD in nats along the channel axis, calculated in FP32."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if left.shape != right.shape or left.ndim < 2 or left.shape[1] < 2:
        raise ValueError('Expected matching tensors with at least two channels')
    lp = F.log_softmax(left.float() / temperature, dim=1)
    lq = F.log_softmax(right.float() / temperature, dim=1)
    lm = torch.logaddexp(lp, lq) - math.log(2)
    return (.5 * (lp.exp() * (lp - lm) + lq.exp() * (lq - lm))).sum(1).clamp_min(0)


class JSDAlignmentLoss(nn.Module):
    def __init__(self, *, temperature: float = 1.0, reduction: str = 'level',
                 mask_padding: bool = True, region: str = 'all',
                 report_mask_stats: bool = False):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        if reduction not in ('level', 'pixel'):
            raise ValueError('reduction must be level or pixel')
        if region not in {'all', 'foreground'}:
            raise ValueError("region must be all or foreground")
        if region == 'foreground' and not mask_padding:
            raise ValueError("foreground alignment requires padding exclusion")
        self.temperature, self.reduction, self.mask_padding = temperature, reduction, mask_padding
        self.region, self.report_mask_stats = region, report_mask_stats

    def forward(self, rgb: Sequence[Tensor], thermal: Sequence[Tensor],
                valid_mask: Tensor | None = None, *,
                targets: list[dict] | None = None, image_hw: tuple[int, int] | None = None):
        if not rgb or len(rgb) != len(thermal):
            raise ValueError('Expected matching nonempty feature pyramids')
        batch = rgb[0].shape[0]
        if self.mask_padding and (valid_mask is None or valid_mask.ndim != 4 or
                                  valid_mask.shape[:2] != (batch, 1)):
            raise ValueError('valid_mask must have shape [B, 1, H, W]')
        if self.region == 'foreground' or self.report_mask_stats:
            if targets is None or image_hw is None or valid_mask is None:
                raise ValueError("GT regions require targets, image_hw, and valid_mask")
        numerators, counts, stats = [], [], {}
        for index, (left, right) in enumerate(zip(rgb, thermal, strict=True)):
            if left.ndim != 4 or left.shape[0] != batch:
                raise ValueError('Each level must be BCHW with the same batch size')
            js = channel_js(left, right, self.temperature)
            mask = (F.interpolate(valid_mask.float(), size=js.shape[-2:], mode='nearest-exact')[:, 0]
                    if self.mask_padding else torch.ones_like(js))
            if self.region == 'foreground' or self.report_mask_stats:
                masks, coverage = region_masks(valid_mask, targets, js.shape[-2:], image_hw,
                                                return_counts=True)
                mask = masks[self.region].to(js.dtype)
                if self.report_mask_stats:
                    level = index + 3
                    stats[f'jsd_p{level}_foreground_fraction'] = (
                        masks['foreground'].sum() / masks['all'].sum().clamp_min(1)).detach()
                    stats[f'jsd_p{level}_selected_cells_per_pair'] = (mask.sum() / batch).detach()
                    stats[f'jsd_p{level}_objects_per_pair'] = (coverage['objects'] / batch).detach()
                    stats[f'jsd_p{level}_covered_objects_per_pair'] = (coverage['covered_objects'] / batch).detach()
            amount, count = (js * mask).sum(), mask.sum()
            numerators.append(amount); counts.append(count)
            stats[f'ife_js_p{index + 3}'] = (amount / count.clamp_min(1)).detach()
        if self.reduction == 'pixel':
            loss = sum(numerators) / sum(counts).clamp_min(1)
        else:
            # Levels with no valid positions must not dilute the others.
            present = torch.stack([count > 0 for count in counts]).sum().clamp_min(1)
            loss = sum(value / count.clamp_min(1) for value, count in zip(numerators, counts)) / present
        return loss, {**stats, 'ife_js': loss.detach(), 'fs': loss.detach()}

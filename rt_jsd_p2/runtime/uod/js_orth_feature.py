"""Align paired IFE with JS and separate each NFE with squared cosine."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class JSOrthFeatureLoss(nn.Module):
    """Pixelwise channel distributions for JS; raw channel vectors for cosine.

    Squared cosine has its minimum at orthogonality (cos=0), not cos=-1.
    By default the orthogonality term treats IFE as a detached reference;
    detection and JS train IFE, while orthogonality trains each matching NFE
    and the shared BFE through that NFE. There is no cross-modal NFE term.
    """

    def __init__(self, *, temperature: float = 1.0, js_weight: float = 1.0,
                 orth_weight: float = 1.0, detach_ife: bool = True,
                 mask_padding: bool = True):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("JS temperature must be finite and positive")
        if any(not math.isfinite(w) or w < 0 for w in (js_weight, orth_weight)):
            raise ValueError("Feature weights must be finite and non-negative")
        self.temperature = temperature
        self.js_weight = js_weight
        self.orth_weight = orth_weight
        self.detach_ife = detach_ife
        self.mask_padding = mask_padding

    def forward(self, intrinsic_rgb: Sequence[Tensor], intrinsic_thermal: Sequence[Tensor],
                non_rgb: Sequence[Tensor], non_thermal: Sequence[Tensor],
                valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        groups = (intrinsic_rgb, intrinsic_thermal, non_rgb, non_thermal)
        if not groups[0] or len({len(group) for group in groups}) != 1:
            raise ValueError("All IFE/NFE pyramids must have the same nonzero number of levels")
        batch = groups[0][0].shape[0]
        if self.mask_padding and (
                valid_mask is None or valid_mask.ndim != 4 or
                valid_mask.shape[:2] != (batch, 1)):
            raise ValueError("mask_padding=True requires valid_mask [batch, 1, height, width]")

        sums: dict[str, Tensor] = {}
        pixel_count = groups[0][0].new_zeros((), dtype=torch.float32)
        for tensors in zip(*groups, strict=True):
            if tensors[0].ndim != 4 or len({tuple(t.shape) for t in tensors}) != 1:
                raise ValueError("IFE/NFE tensors at each level must have identical BCHW shapes")
            if tensors[0].shape[0] != batch or tensors[0].shape[1] < 2:
                raise ValueError("Feature levels need the same batch size and at least two channels")
            ir, it, nr, nt = (t.float() for t in tensors)
            log_p = F.log_softmax(ir / self.temperature, dim=1)
            log_q = F.log_softmax(it / self.temperature, dim=1)
            log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
            js = 0.5 * (log_p.exp() * (log_p - log_m) +
                        log_q.exp() * (log_q - log_m)).sum(dim=1)
            if self.detach_ife:
                ir, it = ir.detach(), it.detach()
            values = {
                "ife_js": js.clamp_min(0),
                "rgb_ife_nfe_orth": F.cosine_similarity(ir, nr, dim=1).square(),
                "thermal_ife_nfe_orth": F.cosine_similarity(it, nt, dim=1).square(),
            }
            if self.mask_padding:
                mask = F.interpolate(valid_mask.float(), size=js.shape[-2:],
                                     mode="nearest-exact")[:, 0].bool().float()
            else:
                mask = torch.ones_like(js)
            pixel_count = pixel_count + mask.sum()
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + (value * mask).sum()

        means = {name: value / pixel_count.clamp_min(1) for name, value in sums.items()}
        means["ife_nfe_orth"] = 0.5 * (
            means["rgb_ife_nfe_orth"] + means["thermal_ife_nfe_orth"])
        loss = self.js_weight * means["ife_js"] + self.orth_weight * means["ife_nfe_orth"]
        return loss, {**{name: value.detach() for name, value in means.items()},
                      "fs": loss.detach()}

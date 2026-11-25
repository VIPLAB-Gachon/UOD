"""Symmetric-KL alignment and separation for paired UOD features.

Each C-dimensional feature vector is interpreted as a channel distribution
through softmax.  RGB/thermal IFE distributions are aligned, while matched
IFE/NFE and RGB/thermal NFE distributions are pushed apart.  A smooth margin
keeps the repulsion bounded and avoids the exact zero-gradient region of a
hard hinge.  This is an experimental objective, not a claim about the UOD
paper's original loss.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SymmetricKLFeatureLoss(nn.Module):
    def __init__(self, *, temperature: float = 1.0, margin: float = 0.1,
                 repulsion_temperature: float = 0.05,
                 align_weight: float = 1.0, ife_nfe_weight: float = 1.0,
                 nfe_nfe_weight: float = 1.0, reduction: str = "pixel",
                 mask_padding: bool = True):
        super().__init__()
        if temperature <= 0 or margin <= 0 or repulsion_temperature <= 0:
            raise ValueError("KL temperature, margin and repulsion temperature must be positive")
        if min(align_weight, ife_nfe_weight, nfe_nfe_weight) < 0:
            raise ValueError("KL feature weights must be non-negative")
        if reduction not in {"pixel", "level"}:
            raise ValueError("KL reduction must be pixel or level")
        self.temperature = temperature
        self.margin = margin
        self.repulsion_temperature = repulsion_temperature
        self.align_weight = align_weight
        self.ife_nfe_weight = ife_nfe_weight
        self.nfe_nfe_weight = nfe_nfe_weight
        self.reduction = reduction
        self.mask_padding = mask_padding

    def _distribution(self, features: Tensor) -> tuple[Tensor, Tensor]:
        log_probs = F.log_softmax(features.float() / self.temperature, dim=1)
        return log_probs, log_probs.exp()

    @staticmethod
    def _symmetric_kl(left: tuple[Tensor, Tensor],
                      right: tuple[Tensor, Tensor]) -> Tensor:
        log_p, p = left
        log_q, q = right
        # 0.5 * (KL(p||q) + KL(q||p)), evaluated per spatial location.
        return (0.5 * ((p - q) * (log_p - log_q)).sum(dim=1)).clamp_min(0)

    def _repulsion(self, distance: Tensor) -> Tensor:
        return self.repulsion_temperature * F.softplus(
            (self.margin - distance) / self.repulsion_temperature)

    def forward(self, intrinsic_rgb: Sequence[Tensor], intrinsic_thermal: Sequence[Tensor],
                non_rgb: Sequence[Tensor], non_thermal: Sequence[Tensor],
                valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        groups = (intrinsic_rgb, intrinsic_thermal, non_rgb, non_thermal)
        if not groups[0] or len({len(group) for group in groups}) != 1:
            raise ValueError("All IFE/NFE pyramids must have the same nonzero number of levels")
        if self.mask_padding and valid_mask is None:
            raise ValueError("mask_padding=True requires a valid image mask")

        sums: dict[str, Tensor] = {}
        denominator: Tensor | float = 0.0
        for tensors in zip(*groups, strict=True):
            if len({tuple(value.shape) for value in tensors}) != 1 or tensors[0].ndim != 4:
                raise ValueError("IFE/NFE tensors at a level must have identical BCHW shapes")
            if tensors[0].shape[1] < 2:
                raise ValueError("Channel softmax KL requires at least two feature channels")
            ir, it, nr, nt = (self._distribution(value) for value in tensors)
            d_align = self._symmetric_kl(ir, it)
            d_rgb = self._symmetric_kl(ir, nr)
            d_thermal = self._symmetric_kl(it, nt)
            d_nfe = self._symmetric_kl(nr, nt)
            values = {
                "align_kl": d_align,
                "rgb_ife_nfe_kl": d_rgb,
                "thermal_ife_nfe_kl": d_thermal,
                "nfe_nfe_kl": d_nfe,
                "rgb_repulsion": self._repulsion(d_rgb),
                "thermal_repulsion": self._repulsion(d_thermal),
                "nfe_repulsion": self._repulsion(d_nfe),
            }
            if self.mask_padding:
                if valid_mask.ndim != 4 or valid_mask.shape[:2] != (tensors[0].shape[0], 1):
                    raise ValueError("valid_mask must have shape [batch, 1, height, width]")
                mask = F.interpolate(valid_mask.float(), size=d_align.shape[-2:],
                                     mode="nearest-exact")[:, 0]
            else:
                mask = torch.ones_like(d_align)
            count = mask.sum().clamp_min(1)
            for key, value in values.items():
                amount = (value * mask).sum()
                if self.reduction == "level":
                    amount = amount / count
                sums[key] = sums.get(key, 0.0) + amount
            denominator = denominator + (count if self.reduction == "pixel" else 1.0)

        averages = {key: value / denominator for key, value in sums.items()}
        loss = (self.align_weight * averages["align_kl"] +
                self.ife_nfe_weight * 0.5 * (
                    averages["rgb_repulsion"] + averages["thermal_repulsion"]) +
                self.nfe_nfe_weight * averages["nfe_repulsion"])
        stats = {key: value.detach() for key, value in averages.items()}
        stats["fs"] = loss.detach()
        return loss, stats

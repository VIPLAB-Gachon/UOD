"""JS-aligned IFE, GT prediction consistency, and modality-aware NFE loss.

This is an experimental objective, not the original UOD paper loss.  The
single-class branch compares YOLO objectness as a [background, object]
distribution; softmax over a single class would otherwise be identically 1.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class JSOrthModalityFeatureLoss(nn.Module):
    """Align paired IFE features, orthogonalize NFE, distinguish NFE modality.

    The modality classifier uses antipodal logits [s, -s], where s is the
    cosine projection onto a learned direction.  RGB is class 0 and thermal
    is class 1.  A paired negative-cosine term additionally encourages their
    complete feature vectors, not just their projections, to point apart.
    """

    def __init__(self, channels: Sequence[int], num_classes: int, *,
                 feature_temperature: float = 1.0, gt_temperature: float = 1.0,
                 modality_temperature: float = 0.2, max_positions: int = 256,
                 js_weight: float = 0.1, gt_kl_weight: float = 0.1,
                 orth_weight: float = 0.05, modality_weight: float = 0.05,
                 opposition_weight: float = 0.05, mask_padding: bool = True):
        super().__init__()
        if len(channels) != 3 or min(channels) < 2 or num_classes < 1:
            raise ValueError("Expected three feature levels, >=2 channels, and >=1 class")
        if min(feature_temperature, gt_temperature, modality_temperature) <= 0:
            raise ValueError("All temperatures must be positive")
        if max_positions < 1:
            raise ValueError("max_positions must be positive")
        weights = (js_weight, gt_kl_weight, orth_weight,
                   modality_weight, opposition_weight)
        if min(weights) < 0:
            raise ValueError("Feature loss weights must be non-negative")
        self.num_classes = num_classes
        self.feature_temperature = feature_temperature
        self.gt_temperature = gt_temperature
        self.modality_temperature = modality_temperature
        self.max_positions = max_positions
        self.weights = weights
        self.mask_padding = mask_padding
        self.modality_direction = nn.ParameterList(
            nn.Parameter(torch.randn(channels_at_level)) for channels_at_level in channels)

    @staticmethod
    def _symmetric_kl(log_p: Tensor, log_q: Tensor) -> Tensor:
        p, q = log_p.exp(), log_q.exp()
        return (0.5 * ((p - q) * (log_p - log_q)).sum(dim=-1)).clamp_min(0)

    def _ife_js(self, rgb: Tensor, thermal: Tensor) -> Tensor:
        log_p = F.log_softmax(rgb.float() / self.feature_temperature, dim=1)
        log_q = F.log_softmax(thermal.float() / self.feature_temperature, dim=1)
        log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
        js = 0.5 * (log_p.exp() * (log_p - log_m) +
                    log_q.exp() * (log_q - log_m)).sum(dim=1)
        return js.clamp_min(0)

    @staticmethod
    def _valid_at_level(valid_mask: Tensor | None, feature: Tensor,
                        mask_padding: bool) -> Tensor:
        if not mask_padding:
            return torch.ones(feature.shape[0], *feature.shape[-2:],
                              device=feature.device, dtype=torch.bool)
        if valid_mask is None or valid_mask.ndim != 4 or valid_mask.shape[:2] != (feature.shape[0], 1):
            raise ValueError("mask_padding=True needs valid_mask [batch, 1, height, width]")
        return F.interpolate(valid_mask.float(), size=feature.shape[-2:],
                             mode="nearest-exact")[:, 0].bool()

    def _gt_consistency(self, pred_rgb: Sequence[Tensor], pred_thermal: Sequence[Tensor],
                        matched_indices: Sequence[tuple[Tensor, Tensor, Tensor, Tensor]],
                        zero: Tensor) -> tuple[Tensor, int]:
        distances = []
        pairs = 0
        for rgb, thermal, (batch, anchor, gy, gx) in zip(
                pred_rgb, pred_thermal, matched_indices, strict=True):
            if rgb.shape != thermal.shape or rgb.ndim != 5 or rgb.shape[-1] != self.num_classes + 5:
                raise ValueError("RGB/thermal YOLO raw predictions must match [B,A,H,W,5+nc]")
            if batch.numel() == 0:
                continue
            raw_r = rgb[batch, anchor, gy, gx].float()
            raw_t = thermal[batch, anchor, gy, gx].float()
            if self.num_classes == 1:
                # Softmax of the sole class would always be [1].  The actual
                # detector's objectness logit defines P(background/object).
                logits_r = torch.stack((torch.zeros_like(raw_r[:, 4]), raw_r[:, 4]), dim=-1)
                logits_t = torch.stack((torch.zeros_like(raw_t[:, 4]), raw_t[:, 4]), dim=-1)
            else:
                logits_r, logits_t = raw_r[:, 5:], raw_t[:, 5:]
            log_p = F.log_softmax(logits_r / self.gt_temperature, dim=-1)
            log_q = F.log_softmax(logits_t / self.gt_temperature, dim=-1)
            distances.append(self._symmetric_kl(log_p, log_q))
            pairs += batch.numel()
        if not distances:
            return zero, 0
        return torch.cat(distances).mean(), pairs

    def _modality_and_opposition(self, rgb: Tensor, thermal: Tensor,
                                 valid: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        batch, channels, height, width = rgb.shape
        flat_rgb = rgb.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        flat_thermal = thermal.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        valid = valid.reshape(batch, height * width)
        direction = F.normalize(self.modality_direction[level].float(), dim=0)
        modality_losses, opposition_losses, paired_cosines = [], [], []
        for image in range(batch):
            positions = valid[image].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            if positions.numel() > self.max_positions:
                take = torch.linspace(0, positions.numel() - 1, self.max_positions,
                                      device=positions.device).long()
                positions = positions[take]
            features_r = F.normalize(flat_rgb[image, positions].float(), dim=-1)
            features_t = F.normalize(flat_thermal[image, positions].float(), dim=-1)
            score_r = (features_r @ direction) / self.modality_temperature
            score_t = (features_t @ direction) / self.modality_temperature
            logits = torch.cat((torch.stack((score_r, -score_r), dim=-1),
                                torch.stack((score_t, -score_t), dim=-1)))
            labels = torch.cat((torch.zeros_like(score_r, dtype=torch.long),
                                torch.ones_like(score_t, dtype=torch.long)))
            modality_losses.append(F.cross_entropy(logits, labels))
            cosine = (features_r * features_t).sum(dim=-1)
            paired_cosines.append(cosine.mean())
            opposition_losses.append(((1.0 + cosine) * 0.5).mean())
        zero = (rgb.float().sum() + thermal.float().sum()) * 0.0
        if not modality_losses:
            return zero, zero, zero
        return (torch.stack(modality_losses).mean(),
                torch.stack(opposition_losses).mean(),
                torch.stack(paired_cosines).mean())

    def forward(self, outputs: dict[str, object],
                matched_indices: Sequence[tuple[Tensor, Tensor, Tensor, Tensor]],
                valid_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        names = ("intrinsic_rgb", "intrinsic_thermal", "non_rgb", "non_thermal")
        groups = tuple(outputs[name] for name in names)
        if any(len(group) != 3 for group in groups) or len(matched_indices) != 3:
            raise ValueError("Expected three YOLO feature and assignment levels")
        zero = groups[0][0].float().sum() * 0.0
        js_sum, orth_sum, pixel_count = zero, zero, zero
        modality_terms, opposition_terms, cosine_terms = [], [], []
        for level, tensors in enumerate(zip(*groups, strict=True)):
            ir, it, nr, nt = tensors
            if len({tuple(value.shape) for value in tensors}) != 1:
                raise ValueError("IFE/NFE features must match at each level")
            valid = self._valid_at_level(valid_mask, ir, self.mask_padding)
            mask = valid.float()
            js = self._ife_js(ir, it)
            # Preserve detector IFE as the semantic anchor; move NFE for orthogonality.
            orth = 0.5 * (F.cosine_similarity(ir.detach().float(), nr.float(), dim=1).square() +
                          F.cosine_similarity(it.detach().float(), nt.float(), dim=1).square())
            js_sum = js_sum + (js * mask).sum()
            orth_sum = orth_sum + (orth * mask).sum()
            pixel_count = pixel_count + mask.sum()
            modality, opposition, cosine = self._modality_and_opposition(nr, nt, valid, level)
            modality_terms.append(modality)
            opposition_terms.append(opposition)
            cosine_terms.append(cosine)

        ife_js = js_sum / pixel_count.clamp_min(1)
        orthogonal = orth_sum / pixel_count.clamp_min(1)
        gt_kl, gt_pairs = self._gt_consistency(
            outputs["pred_rgb"], outputs["pred_thermal"], matched_indices, zero)
        modality_ce = torch.stack(modality_terms).mean()
        nfe_opposition = torch.stack(opposition_terms).mean()
        nfe_cosine = torch.stack(cosine_terms).mean()
        js_w, gt_w, orth_w, modality_w, opposition_w = self.weights
        loss = (js_w * ife_js + gt_w * gt_kl + orth_w * orthogonal +
                modality_w * modality_ce + opposition_w * nfe_opposition)
        stats = {
            "ife_js": ife_js.detach(),
            "gt_symmetric_kl": gt_kl.detach(),
            "gt_pairs": ife_js.new_tensor(gt_pairs).detach(),
            "ife_nfe_orth": orthogonal.detach(),
            "modality_ce": modality_ce.detach(),
            "nfe_opposition": nfe_opposition.detach(),
            "nfe_pair_cosine": nfe_cosine.detach(),
            "fs": loss.detach(),
        }
        return loss, stats

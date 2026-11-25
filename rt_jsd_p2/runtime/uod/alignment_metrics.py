"""Held-out alignment, feature-collapse, and object/background diagnostics.

Foreground is a feature-cell center inside a GT box, not a segmentation label.
The linear probe uses class centroids fit ONLY on an unaugmented training subset.
Its ROC AUC on held-out data complements, and does not replace, detection mAP.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import roi_align
from .js_alignment import channel_js
from .feature_regions import region_masks


def evenly_spaced_indices(total: int, limit: int) -> list[int]:
    if total < 1 or limit < 1:
        raise ValueError('total and limit must be positive')
    count = min(total, limit)
    return [int(x) for x in torch.linspace(0, total - 1, count).round().tolist()]


def sampled_vectors(feature: Tensor, mask: Tensor, limit: int) -> Tensor:
    """Deterministic spatial sampling, bounded separately per image and class."""
    values = []
    for image, image_mask in zip(feature, mask, strict=True):
        indices = image_mask.flatten().nonzero().flatten()
        if indices.numel() > limit:
            indices = indices[torch.linspace(0, indices.numel() - 1, limit,
                                            device=indices.device).round().long()]
        values.append(image.flatten(1)[:, indices].T)
    return torch.cat(values)


def binary_auc(positive: Tensor, negative: Tensor) -> float | None:
    """Exact ROC AUC with half credit for tied scores; no sklearn dependency."""
    positive, negative = positive.double().cpu(), negative.double().cpu()
    if not len(positive) or not len(negative):
        return None
    sorted_negative = negative.sort().values
    lo = torch.searchsorted(sorted_negative, positive, right=False)
    hi = torch.searchsorted(sorted_negative, positive, right=True)
    return float((lo.double() + .5 * (hi - lo)).sum() / (len(positive) * len(negative)))


class ForegroundCentroidProbe:
    """Linear direction separating class centroids of L2-normalized features."""
    def __init__(self, samples_per_region: int = 128):
        if samples_per_region < 1:
            raise ValueError('samples_per_region must be positive')
        self.limit = samples_per_region
        self.sums: dict[tuple, Tensor] = {}
        self.counts: dict[tuple, int] = {}

    @torch.no_grad()
    def update(self, rgb, thermal, valid_mask, targets, image_hw):
        for level, (left, right) in enumerate(zip(rgb, thermal, strict=True), 3):
            masks = region_masks(valid_mask, targets, left.shape[-2:], image_hw)
            for mode, feature in (('rgb', left), ('thermal', right)):
                for region in ('foreground', 'background'):
                    rows = sampled_vectors(feature.float(), masks[region], self.limit)
                    rows = F.normalize(rows, dim=1)
                    key = (level, mode, region)
                    value = rows.double().sum(0)
                    self.sums[key] = self.sums.get(key, torch.zeros_like(value)) + value
                    self.counts[key] = self.counts.get(key, 0) + len(rows)

    def direction(self, level: int, mode: str) -> Tensor | None:
        fg, bg = (level, mode, 'foreground'), (level, mode, 'background')
        if not self.counts.get(fg) or not self.counts.get(bg):
            return None
        return (self.sums[fg] / self.counts[fg] - self.sums[bg] / self.counts[bg]).float()

    def state_dict(self):
        return {f'P{level}/{mode}/{region}': {'count': count,
                    'sum': self.sums[(level, mode, region)].cpu()}
                for (level, mode, region), count in self.counts.items()}


@torch.no_grad()
def diagnostic_roi_vectors(feature, boxes):
    """Fixed-grid ROIAlign forward only, without torchvision's compiled backward path.

    The native forward computes each output independently. No ROIAlign backward
    is used for this diagnostic; the nondeterministic backward is irrelevant.
    Calling the native op also avoids compiling a new graph for every GT count
    when global deterministic algorithms are enabled.
    """
    rois = torch.cat([torch.cat((box.new_full((len(box), 1), index), box), 1)
                      for index, box in enumerate(boxes)], 0)
    return torch.ops.torchvision.roi_align(feature, rois, 1.0, 3, 3, 2, True).mean((2, 3))


class AlignmentDiagnostics:
    def __init__(self, temperature: float = 1.0, probe: ForegroundCentroidProbe | None = None,
                 samples_per_region: int = 128):
        if not math.isfinite(temperature) or temperature <= 0 or samples_per_region < 1:
            raise ValueError('Invalid diagnostic temperature/sample limit')
        self.temperature, self.probe, self.limit = temperature, probe, samples_per_region
        self.totals, self.counts, self.moments, self.scores = {}, {}, {}, {}
        self.pairs, self.objects = 0, 0

    def _record(self, key, values: dict[str, Tensor], mask: Tensor):
        count = mask.sum().double()
        self.counts[key] = self.counts.get(key, torch.zeros_like(count)) + count
        for name, value in values.items():
            full_key = (*key, name)
            amount = value[mask].double().sum()
            self.totals[full_key] = self.totals.get(full_key, torch.zeros_like(amount)) + amount

    @torch.no_grad()
    def update(self, rgb, thermal, valid_mask, targets, image_hw):
        self.pairs += len(targets)
        self.objects += sum(len(t['boxes']) for t in targets)
        for level, (left, right) in enumerate(zip(rgb, thermal, strict=True), 3):
            left, right = left.float(), right.float()
            masks = region_masks(valid_mask, targets, left.shape[-2:], image_hw)
            values = {'js': channel_js(left, right, self.temperature),
                      'cosine': F.cosine_similarity(left, right, dim=1)}
            for mode, feature in (('rgb', left), ('thermal', right)):
                lp = F.log_softmax(feature / self.temperature, dim=1)
                values[f'{mode}_entropy'] = -(lp.exp() * lp).sum(1) / math.log(feature.shape[1])
                values[f'{mode}_rms'] = feature.square().mean(1).sqrt()
            for region, mask in masks.items():
                self._record((level, region), values, mask)
                for mode, feature in (('rgb', left), ('thermal', right)):
                    # Sampled channel moments avoid storing full feature maps.
                    rows = sampled_vectors(feature, mask, self.limit).double()
                    key = (level, region, mode)
                    n, total, square = self.moments.get(key, (0, torch.zeros(feature.shape[1], device=feature.device, dtype=torch.float64), torch.zeros(feature.shape[1], device=feature.device, dtype=torch.float64)))
                    self.moments[key] = (n + len(rows), total + rows.sum(0), square + rows.square().sum(0))
                    if self.probe is not None and region != 'all':
                        direction = self.probe.direction(level, mode)
                        if direction is not None:
                            score = F.normalize(rows.float(), dim=1) @ direction.to(feature.device)
                            self.scores.setdefault(key, []).append(score.cpu())
            # Each GT box contributes one paired ROI vector, including tiny boxes
            # whose interiors do not contain a feature-cell center.
            scale_y = left.shape[-2] / image_hw[0]
            scale_x = left.shape[-1] / image_hw[1]
            boxes = [t['boxes'].float() * left.new_tensor([scale_x, scale_y, scale_x, scale_y]) for t in targets]
            if sum(len(box) for box in boxes):
                rois_l = diagnostic_roi_vectors(left, boxes)
                rois_r = diagnostic_roi_vectors(right, boxes)
                self._record((level, 'objects'),
                    {'js': channel_js(rois_l, rois_r, self.temperature),
                     'cosine': F.cosine_similarity(rois_l, rois_r, dim=1)},
                    torch.ones(len(rois_l), device=left.device, dtype=torch.bool))

    def compute(self):
        levels = {}
        for level in sorted({key[0] for key in self.counts}):
            regions = {}
            for (lev, region), count in self.counts.items():
                if lev != level:
                    continue
                n = int(count)
                item = {'count': n}
                for (ll, rr, name), amount in self.totals.items():
                    if (ll, rr) == (level, region):
                        item[name] = float(amount / count) if n else None
                for mode in ('rgb', 'thermal'):
                    moments = self.moments.get((level, region, mode))
                    if moments is not None:
                        size, total, square = moments
                        item[f'{mode}_channel_variance'] = float((square / size - (total / size).square()).clamp_min(0).mean()) if size else None
                        item[f'{mode}_variance_samples'] = size
                regions[region] = item
            for mode in ('rgb', 'thermal'):
                positive = self.scores.get((level, 'foreground', mode), [])
                negative = self.scores.get((level, 'background', mode), [])
                regions[f'{mode}_probe'] = {
                    'auc': binary_auc(torch.cat(positive), torch.cat(negative)) if positive and negative else None,
                    'foreground_samples': sum(len(v) for v in positive),
                    'background_samples': sum(len(v) for v in negative),
                }
            levels[f'P{level}'] = regions
        return {'pairs': self.pairs, 'objects': self.objects, 'temperature': self.temperature,
                'levels': levels,
                'probe': 'Training-subset object/background centroid direction on normalized features; held-out ROC AUC. GT-box interior is a coarse foreground label.',
                'variance': 'Mean per-channel variance across deterministic sampled feature vectors.',
                'objects_metric': 'JSD/cosine between paired GT ROIAlign(3x3)-pooled feature vectors; each object has equal weight.'}

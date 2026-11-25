"""Signed boundary distributions used by both training and actual inference.

Distances are L,T,R,B in anchor-width/height units. Signed distances retain
YOLOv5 neighbour-cell positives whose reference point lies outside the box.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_YOLO_ROOT = Path(__file__).resolve().parents[3] / "rt_jsd" / "runtime" / "third_party" / "yolov5"
if str(_YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(_YOLO_ROOT))
from models.yolo import Detect  # noqa: E402


def _support_for(logits: Tensor, support: Tensor) -> Tensor:
    if logits.ndim < 2 or logits.shape[-2] != 4:
        raise ValueError("Expected logits shaped [...,4,K]")
    if support.ndim != 1 or support.numel() < 2:
        raise ValueError("support must be a one-dimensional tensor with >=2 bins")
    if logits.shape[-1] != support.numel():
        raise ValueError("logit bin count does not match support")
    return support.to(device=logits.device, dtype=torch.float32)


def decode_distribution(logits: Tensor, points: Tensor, anchor_wh: Tensor,
                        support: Tensor) -> tuple[Tensor, Tensor]:
    """Return actual xywh boxes and a mask of valid *unclamped* box sizes.

    points and anchor_wh end in 2 and broadcast against logits[..., 0, 0].
    Their units determine the output units. Softmax/expectation uses FP32.
    For inverted corners, preserve the midpoint and clamp only width/height;
    the original signed DFL targets still supervise all four distributions.
    """
    support = _support_for(logits, support)
    if points.shape[-1] != 2 or anchor_wh.shape[-1] != 2:
        raise ValueError("points and anchor_wh must end in two coordinates")
    distances = (logits.float().softmax(-1) * support).sum(-1)
    points = points.to(device=logits.device, dtype=torch.float32)
    anchor_wh = anchor_wh.to(device=logits.device, dtype=torch.float32)
    left, top, right, bottom = distances.unbind(-1)
    aw, ah = anchor_wh.unbind(-1)
    cx = points[..., 0] + (right - left) * aw * .5
    cy = points[..., 1] + (bottom - top) * ah * .5
    width, height = (left + right) * aw, (top + bottom) * ah
    valid = ((width > 0) & (height > 0) & torch.isfinite(cx)
             & torch.isfinite(cy) & torch.isfinite(width) & torch.isfinite(height))
    return torch.stack((cx, cy, width.clamp_min(1e-6), height.clamp_min(1e-6)), -1), valid


def distribution_focal_loss_per_edge(logits: Tensor, target_distances: Tensor,
                                     support: Tensor) -> Tensor:
    """Two-adjacent-bin GT cross entropy for each of the four edges.

    Return shape is logits.shape[:-1]. Targets outside support are errors;
    silently clipping a signed target would change the ground-truth box.
    """
    support = _support_for(logits, support)
    if target_distances.shape != logits.shape[:-1]:
        raise ValueError("target_distances must have shape logits.shape[:-1]")
    steps = support[1:] - support[:-1]
    if not bool(torch.isfinite(support).all()) or not bool((steps > 0).all()):
        raise ValueError("support must be finite and strictly increasing")
    if not torch.allclose(steps, steps[:1].expand_as(steps), rtol=1e-5, atol=1e-6):
        raise ValueError("DFL requires uniformly spaced support")
    target = target_distances.to(device=logits.device, dtype=torch.float32)
    invalid = ~torch.isfinite(target) | (target < support[0]) | (target > support[-1])
    if bool(invalid.any()):
        raise ValueError("GT signed distance is nonfinite or outside the distribution support")
    coordinate = (target - support[0]) / steps[0]
    lower = coordinate.floor().long().clamp(max=support.numel() - 1)
    upper = (lower + 1).clamp(max=support.numel() - 1)
    upper_weight = coordinate - lower.float()
    log_probs = F.log_softmax(logits.float(), dim=-1)
    lower_log = log_probs.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    upper_log = log_probs.gather(-1, upper.unsqueeze(-1)).squeeze(-1)
    return -(1 - upper_weight) * lower_log - upper_weight * upper_log


def distribution_focal_loss(logits: Tensor, target_distances: Tensor,
                            support: Tensor) -> Tensor:
    """Original API: average two-bin GT cross entropy over four edges."""
    return distribution_focal_loss_per_edge(logits, target_distances, support).mean(-1)


def directed_jsd(student_logits: Tensor, teacher_logits: Tensor,
                 temperature: float = 1.0) -> Tensor:
    """JSD with teacher detached, averaged over four edges (no T**2 factor).

    The mixture retains its student dependency. Both inputs are [...,4,K].
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if student_logits.ndim < 2 or student_logits.shape[-2] != 4:
        raise ValueError("Expected logits shaped [...,4,K]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    log_student = F.log_softmax(student_logits.float() / temperature, -1)
    log_teacher = F.log_softmax(teacher_logits.detach().float() / temperature, -1)
    log_mixture = torch.logaddexp(log_student, log_teacher) - math.log(2.0)
    divergence = .5 * (
        (log_student.exp() * (log_student - log_mixture)).sum(-1)
        + (log_teacher.exp() * (log_teacher - log_mixture)).sum(-1))
    return divergence.mean(-1).clamp_min(0.0)


def directed_jsd_per_edge(student_logits: Tensor, teacher_logits: Tensor,
                 temperature: float = 1.0) -> Tensor:
    """JSD with teacher detached for each edge (no T**2 factor).

    The mixture retains its student dependency. Both inputs are [...,4,K].
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if student_logits.ndim < 2 or student_logits.shape[-2] != 4:
        raise ValueError("Expected logits shaped [...,4,K]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    log_student = F.log_softmax(student_logits.float() / temperature, -1)
    log_teacher = F.log_softmax(teacher_logits.detach().float() / temperature, -1)
    log_mixture = torch.logaddexp(log_student, log_teacher) - math.log(2.0)
    divergence = .5 * (
        (log_student.exp() * (log_student - log_mixture)).sum(-1)
        + (log_teacher.exp() * (log_teacher - log_mixture)).sum(-1))
    return divergence.clamp_min(0.0)


class DistributionDetect(Detect):
    """Replace a YOLOv5 Detect box projection with GT-supervised distributions.

    Objectness/class weights are copied per anchor. Distribution weights start
    at zero, with a Gaussian bias around a 0.5-anchor distance on each edge.
    Training returns raw distribution logits; evaluation returns standard
    YOLO decoded xywh/objectness/class outputs plus the same raw logits.
    """

    def __init__(self, base: Detect, bins: int = 25, support_min: float = -2.,
                 support_max: float = 4., decode_chunk_size: int = 32768):
        nn.Module.__init__(self)
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("bins must be an integer >=2")
        if (not math.isfinite(support_min) or not math.isfinite(support_max)
                or support_min >= .5 or support_max <= .5):
            raise ValueError("finite support must surround initial distance 0.5")
        if decode_chunk_size < 1:
            raise ValueError("decode_chunk_size must be positive")
        if isinstance(base, DistributionDetect):
            raise ValueError("base must be a continuous-box YOLOv5 Detect head")
        self.nc, self.nl, self.na = base.nc, base.nl, base.na
        self.bins = bins
        self.reg_max = bins - 1
        self.no = 4 * bins + 1 + self.nc
        self.decoded_no = 5 + self.nc
        self.decode_chunk_size = int(decode_chunk_size)
        self.inplace = base.inplace
        self.dynamic = base.dynamic
        self.export = base.export
        self.stride = base.stride.detach().clone()
        self.register_buffer("anchors", base.anchors.detach().clone())
        self.register_buffer("support", torch.linspace(
            support_min, support_max, bins, device=base.anchors.device))
        self.grid = [torch.empty(0, device=base.anchors.device) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0, device=base.anchors.device) for _ in range(self.nl)]
        self.m = nn.ModuleList()
        for old in base.m:
            new = nn.Conv2d(old.in_channels, self.na * self.no, 1,
                            device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                new.weight.zero_()
                new.bias.zero_()
                old_w = old.weight.view(self.na, self.nc + 5, old.in_channels, 1, 1)
                new_w = new.weight.view(self.na, self.no, old.in_channels, 1, 1)
                new_w[:, 4 * bins:].copy_(old_w[:, 4:])
                bias = new.bias.view(self.na, self.no)
                if old.bias is not None:
                    bias[:, 4 * bins:].copy_(old.bias.view(self.na, self.nc + 5)[:, 4:])
                gaussian = -.5 * ((self.support - .5) / .5).square()
                bias[:, :4 * bins].copy_(gaussian.repeat(4).to(bias.dtype))
            self.m.append(new)
        for name in ("i", "f", "type"):
            if hasattr(base, name):
                setattr(self, name, getattr(base, name))
        self.np = sum(parameter.numel() for parameter in self.parameters())
        self.train(base.training)

    def decode_selected(self, raw_selected: Tensor, level: int,
                        anchor_ids: Tensor, grid_xy: Tensor) -> tuple[Tensor, Tensor]:
        """Decode selected raw rows into pixels, using (grid_x,grid_y)."""
        logits = raw_selected[..., :4 * self.bins].reshape(*raw_selected.shape[:-1], 4, self.bins)
        stride = self.stride[level].float()
        points = (grid_xy.float() + .5) * stride
        anchor_wh = self.anchors[level][anchor_ids].float() * stride
        return decode_distribution(logits, points, anchor_wh, self.support)

    def _decode_level(self, raw: Tensor, level: int) -> Tensor:
        batch, _, ny, nx, _ = raw.shape
        if self.dynamic or self.grid[level].shape[2:4] != raw.shape[2:4]:
            self.grid[level], self.anchor_grid[level] = self._make_grid(nx, ny, level)
        points = ((self.grid[level].float() + 1.) * self.stride[level].float()).reshape(-1, 2)
        anchors = self.anchor_grid[level].float().reshape(-1, 2)
        flat = raw.reshape(batch, -1, self.no)
        decoded_batch = []
        for image_raw in flat:
            decoded_chunks = []
            for start in range(0, image_raw.shape[0], self.decode_chunk_size):
                end = min(start + self.decode_chunk_size, image_raw.shape[0])
                selected = image_raw[start:end]
                logits = selected[:, :4 * self.bins].reshape(-1, 4, self.bins)
                boxes, _ = decode_distribution(logits, points[start:end], anchors[start:end], self.support)
                scores = selected[:, 4 * self.bins:].float().sigmoid()
                decoded_chunks.append(torch.cat((boxes, scores), -1))
            decoded_batch.append(torch.cat(decoded_chunks, 0))
        return torch.stack(decoded_batch, 0)

    def forward(self, features: list[Tensor]):
        raw_levels = []
        decoded_levels = []
        for level, feature in enumerate(features):
            projected = self.m[level](feature)
            batch, _, ny, nx = projected.shape
            raw = projected.view(batch, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw_levels.append(raw)
            if not self.training:
                decoded_levels.append(self._decode_level(raw, level))
        if self.training:
            return raw_levels
        decoded = torch.cat(decoded_levels, 1)
        return (decoded,) if self.export else (decoded, raw_levels)

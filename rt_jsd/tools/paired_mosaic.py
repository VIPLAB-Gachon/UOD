"""Training-only four-image Mosaic for an already validated paired dataset.

The RGB, thermal, validity mask and boxes use the same sampled geometry.
The wrapped dataset supplies the train manifest and original, unaugmented
files; its __getitem__ is used only when Mosaic is inactive. In particular,
validation/test datasets are returned unchanged by ``wrap_dataset``.

Call set_epoch with a ONE-BASED epoch before creating the DataLoader iterator.
Workers must not persist between epochs (the project's loader already uses
persistent_workers=False). No sampler, validation split or base data file is
modified by this wrapper.
"""
from __future__ import annotations

import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from uod.data import read_labels, read_rgb, read_thermal


def _warp_crop(paired, mask, boxes, size, *, scale, dx, dy, fill,
               min_box_size, min_area_ratio):
    """Map a 2S canvas to S using continuous pixel-edge box coordinates.

    x_out = (x_canvas - S)*scale + S/2 + dx; likewise for y.
    A shared inverse sampling grid applies the same map to all four channels.
    Retention is clipped area / full affine-transformed area, so zoom-out alone
    does not penalize a box. dx/dy are pixels in the final S-sized image.
    """
    if paired.shape[-2:] != (2 * size, 2 * size):
        raise ValueError("Mosaic canvas must have spatial dimensions 2S x 2S")
    if mask.shape != (1, 2 * size, 2 * size):
        raise ValueError("Mosaic validity mask must have shape 1 x 2S x 2S")
    if not all(math.isfinite(float(v)) for v in (scale, dx, dy)) or scale <= 0:
        raise ValueError("Invalid Mosaic affine parameters")
    theta = paired.new_tensor([
        [1 / (2 * scale), 0, -dx / (size * scale)],
        [0, 1 / (2 * scale), -dy / (size * scale)],
    ])[None]
    grid = F.affine_grid(theta, (1, paired.shape[0], size, size), align_corners=False)
    output = F.grid_sample((paired - fill)[None], grid, mode="bilinear",
                           padding_mode="zeros", align_corners=False)[0] + fill
    valid = F.grid_sample(mask.float()[None], grid, mode="nearest",
                          padding_mode="zeros", align_corners=False)[0] > .5
    transformed = boxes.clone()
    transformed[:, 0::2] = (transformed[:, 0::2] - size) * scale + size / 2 + dx
    transformed[:, 1::2] = (transformed[:, 1::2] - size) * scale + size / 2 + dy
    full_wh = (transformed[:, 2:] - transformed[:, :2]).clamp_min(0)
    clipped = transformed.clone().clamp_(0, size)
    kept_wh = (clipped[:, 2:] - clipped[:, :2]).clamp_min(0)
    retention = kept_wh.prod(1) / full_wh.prod(1).clamp_min(1e-12)
    keep = (kept_wh[:, 0] > min_box_size) & (kept_wh[:, 1] > min_box_size)
    keep &= retention >= min_area_ratio
    return output, valid, clipped[keep], keep, {
        "scale": float(scale), "dx_pixels": float(dx), "dy_pixels": float(dy),
        "boxes_before": len(boxes), "boxes_after": int(keep.sum()),
        "boxes_clipped": int((clipped != transformed).any(1).sum()),
    }


class PairedMosaicDataset(Dataset):
    """Proxy preserving .samples/.manifest and the uod.data sample schema."""

    def __init__(self, dataset, cfg):
        if not dataset.augment:
            raise ValueError("PairedMosaicDataset is training-only")
        a = cfg.get("augmentation", {})
        self._dataset = dataset
        self.probability = float(a.get("mosaic_probability", 0.))
        self.total_epochs = int(cfg["training"]["epochs"])
        self.close_epochs = int(a.get("mosaic_close_epochs", 10))
        self.scale_gain = float(a.get("scale_gain", getattr(dataset, "scale_gain", 0.)))
        self.translate_fraction = float(a.get("translate_fraction", getattr(dataset, "translate_fraction", 0.)))
        self.min_box_size = float(a.get("min_box_size", getattr(dataset, "min_box_size", 2.)))
        self.min_area_ratio = float(a.get("min_area_ratio", getattr(dataset, "min_area_ratio", .2)))
        self.flip_probability = float(a.get("flip_probability", dataset.flip_probability))
        self.rgb_brightness = float(a.get("rgb_brightness", dataset.rgb_brightness))
        values = (self.probability, self.scale_gain, self.translate_fraction,
                  self.min_box_size, self.min_area_ratio, self.flip_probability, self.rgb_brightness)
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Mosaic augmentation values must be finite")
        if not 0 <= self.probability <= 1 or not 0 <= self.close_epochs <= self.total_epochs:
            raise ValueError("Invalid Mosaic probability or closing epoch count")
        if self.total_epochs < 1 or not 0 <= self.scale_gain < 1 or not 0 <= self.translate_fraction <= 1:
            raise ValueError("Invalid epoch count or Mosaic affine range")
        if self.min_box_size < 0 or not 0 <= self.min_area_ratio <= 1:
            raise ValueError("Invalid Mosaic box retention thresholds")
        if not 0 <= self.flip_probability <= 1 or not 0 <= self.rgb_brightness <= 1:
            raise ValueError("Invalid Mosaic flip/brightness parameters")
        configured_train = cfg.get("data", {}).get("train_manifest")
        if configured_train and Path(dataset.manifest).resolve() != Path(configured_train).resolve():
            raise ValueError("Mosaic may draw only from the configured TRAIN manifest")
        self.epoch = 1

    def __getattr__(self, name):
        # Explicit guard also allows safe spawn/pickle reconstruction.
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __len__(self):
        return len(self._dataset)

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or int(epoch) != epoch or not 1 <= epoch <= self.total_epochs:
            raise ValueError(f"Expected a one-based epoch in [1, {self.total_epochs}]")
        self.epoch = int(epoch)

    @property
    def mosaic_enabled(self):
        return self.probability > 0 and self.epoch <= self.total_epochs - self.close_epochs

    def __getitem__(self, index):
        if not self.mosaic_enabled or random.random() >= self.probability:
            # Base loader applies its existing flip/scale/translation once.
            item = self._dataset[index]
            item["meta"] = {**item["meta"], "mosaic": False, "mosaic_epoch": self.epoch}
            return item

        size = self.image_size
        center_x, center_y = (int(random.uniform(size / 2, size * 3 / 2)) for _ in range(2))
        indices = [index] + random.choices(range(len(self)), k=3)
        random.shuffle(indices)
        canvas = torch.full((4, 2 * size, 2 * size), self.fill, dtype=torch.float32)
        mask = torch.zeros((1, 2 * size, 2 * size), dtype=torch.bool)
        all_boxes, all_labels, sources = [], [], []
        for quadrant, source_index in enumerate(indices):
            sample = self.samples[source_index]
            rgb = read_rgb(sample["rgb"])
            thermal = read_thermal(sample["thermal"], self.thermal_divisor)
            if rgb.shape[0] != 3 or thermal.shape[0] != 1 or rgb.shape[-2:] != thermal.shape[-2:]:
                raise ValueError(f"Mosaic requires an aligned RGB3/T1 pair: {sample['id']}")
            h0, w0 = rgb.shape[-2:]
            boxes, labels = read_labels(Path(sample["label"]), (h0, w0), self.num_classes)
            ratio = size / max(h0, w0)
            w, h = max(1, round(w0 * ratio)), max(1, round(h0 * ratio))
            paired = F.interpolate(torch.cat((rgb, thermal), 0)[None], size=(h, w),
                                   mode="bilinear", align_corners=False)[0]
            if quadrant == 0:  # top-left, anchored at its bottom-right
                x1, y1, x2, y2 = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
                sx1, sy1, sx2, sy2 = w - (x2 - x1), h - (y2 - y1), w, h
            elif quadrant == 1:  # top-right
                x1, y1, x2, y2 = center_x, max(center_y - h, 0), min(center_x + w, 2 * size), center_y
                sx1, sy1, sx2, sy2 = 0, h - (y2 - y1), x2 - x1, h
            elif quadrant == 2:  # bottom-left
                x1, y1, x2, y2 = max(center_x - w, 0), center_y, center_x, min(center_y + h, 2 * size)
                sx1, sy1, sx2, sy2 = w - (x2 - x1), 0, w, y2 - y1
            else:  # bottom-right
                x1, y1, x2, y2 = center_x, center_y, min(center_x + w, 2 * size), min(center_y + h, 2 * size)
                sx1, sy1, sx2, sy2 = 0, 0, x2 - x1, y2 - y1
            canvas[:, y1:y2, x1:x2] = paired[:, sy1:sy2, sx1:sx2]
            mask[:, y1:y2, x1:x2] = True
            boxes[:, 0::2] = boxes[:, 0::2] * (w / w0) + x1 - sx1
            boxes[:, 1::2] = boxes[:, 1::2] * (h / h0) + y1 - sy1
            all_boxes.append(boxes)
            all_labels.append(labels)
            sources.append({"id": sample["id"], "index": int(source_index),
                            "rgb_path": sample["rgb"], "thermal_path": sample["thermal"],
                            "original_hw": [h0, w0], "resized_hw": [h, w],
                            "canvas_xyxy": [x1, y1, x2, y2]})

        boxes, labels = torch.cat(all_boxes, 0), torch.cat(all_labels, 0)
        scale = random.uniform(1 - self.scale_gain, 1 + self.scale_gain)
        dx = random.uniform(-self.translate_fraction, self.translate_fraction) * size
        dy = random.uniform(-self.translate_fraction, self.translate_fraction) * size
        output, valid, boxes, keep, affine = _warp_crop(
            canvas, mask, boxes, size, scale=scale, dx=dx, dy=dy, fill=self.fill,
            min_box_size=self.min_box_size, min_area_ratio=self.min_area_ratio)
        labels = labels[keep]
        flipped = random.random() < self.flip_probability
        if flipped:
            output, valid = output.flip(-1), valid.flip(-1)
            x1 = boxes[:, 0].clone()
            boxes[:, 0] = size - boxes[:, 2]
            boxes[:, 2] = size - x1
        rgb, thermal = output[:3], output[3:]
        if self.rgb_brightness:
            rgb = (rgb * random.uniform(1 - self.rgb_brightness, 1 + self.rgb_brightness)).clamp(0, 1)
        # A composite has no single source-image coordinate frame. For this
        # TRAIN sample its original_target is explicitly the composite frame;
        # validation/test never take this branch and retain source coordinates.
        primary = self.samples[index]
        meta = {"id": primary["id"], "rgb_path": primary["rgb"], "thermal_path": primary["thermal"],
                "original_hw": [size, size], "scale_xy": [1., 1.], "pad_xy": [0, 0],
                "resized_hw": [size, size], "original_target_frame": "augmented_composite",
                "mosaic": True, "mosaic_epoch": self.epoch, "mosaic_sources": sources,
                "mosaic_center_xy": [center_x, center_y], "affine": affine, "flipped": flipped}
        return {"rgb": rgb, "thermal": thermal, "valid_mask": valid,
                "target": {"boxes": boxes, "labels": labels},
                "original_target": {"boxes": boxes.clone(), "labels": labels.clone()}, "meta": meta}


def wrap_dataset(dataset, cfg):
    """Return validation/disabled datasets verbatim; otherwise wrap once."""
    probability = float(cfg.get("augmentation", {}).get("mosaic_probability", 0.))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("mosaic_probability must be finite and in [0,1]")
    if not getattr(dataset, "augment", False) or probability == 0:
        return dataset
    if isinstance(dataset, PairedMosaicDataset):
        return dataset
    return PairedMosaicDataset(dataset, cfg)

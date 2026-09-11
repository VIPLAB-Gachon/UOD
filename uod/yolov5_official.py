"""UOD built from the pinned, official YOLOv5 v7.0 model and loss.

The YOLOv5 detector is split at its natural boundaries:
backbone (BFE, layers 0..9), neck (IFE, layers 10..23), and Detect (IFD,
layer 24).  The two NFE branches are independent copies of the pretrained
neck.  This module intentionally lives beside the smaller reference model so
existing checkpoints and tests keep their original meaning.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Iterable

import torch
from torch import Tensor, nn

from .losses import FeatureSeparationLoss


YOLOV5_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "yolov5"
if not (YOLOV5_ROOT / "models" / "yolo.py").is_file():
    raise ImportError(f"Pinned YOLOv5 source is missing: {YOLOV5_ROOT}")
if str(YOLOV5_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOV5_ROOT))

# YOLOv5 v7.0 uses top-level `models` and `utils` imports internally.
from models.yolo import DetectionModel, Detect  # noqa: E402
from utils.general import non_max_suppression  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.torch_utils import ModelEMA, smart_optimizer  # noqa: E402


def ensure_three_channels(x: Tensor) -> Tensor:
    if x.ndim != 4 or x.shape[1] not in (1, 3):
        raise ValueError(f"Expected BCHW with 1 or 3 channels, got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError("YOLOv5 inputs must be floating point in [0,1]")
    if min(x.shape[-2:]) < 64 or any(v % 32 for v in x.shape[-2:]):
        raise ValueError("Input height/width must be multiples of 32 and at least 64")
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class YOLOv5Stage(nn.Module):
    """Execute a contiguous range of YOLOv5 layers with global route indices."""

    def __init__(self, layers: Iterable[nn.Module], output_indices: tuple[int, ...]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output_indices = output_indices
        layer_indices = {int(m.i) for m in self.layers}
        if any(index not in layer_indices for index in output_indices):
            raise ValueError("Stage output index does not belong to this stage")

    def forward(self, x: Tensor, incoming: dict[int, Tensor] | None = None):
        cache = dict(incoming or {})
        for module in self.layers:
            source = module.f
            if source == -1:
                value = x
            elif isinstance(source, int):
                value = cache[source]
            else:
                value = [x if index == -1 else cache[index] for index in source]
            x = module(value)
            # Only routed tensors and requested outputs are retained. Keeping
            # every early feature would be prohibitively expensive at 1280px.
            if module.i in {4, 6, 10, 14, 17, 20, 23}:
                cache[int(module.i)] = x
        return x, tuple(cache[index] for index in self.output_indices), cache


class OfficialYOLOv5UOD(nn.Module):
    """Pretrained YOLOv5 BFE/IFE/IFD with two training-only NFE necks."""

    def __init__(self, detector: DetectionModel, class_names: list[str],
                 source: dict[str, object], nfe_noise: float = 0.05):
        super().__init__()
        layers = list(detector.model.children())
        if len(layers) != 25 or not isinstance(layers[-1], Detect):
            raise ValueError("Expected the 25-layer YOLOv5 v6/v7 detection topology")
        if [int(m.i) for m in layers] != list(range(25)):
            raise ValueError("Unexpected YOLOv5 layer indices")
        self.bfe = YOLOv5Stage(layers[:10], (4, 6))
        self.ife = YOLOv5Stage(layers[10:24], (17, 20, 23))
        self.ifd = layers[24]
        self.nfe_rgb = copy.deepcopy(self.ife)
        self.nfe_thermal = copy.deepcopy(self.ife)
        if nfe_noise <= 0:
            raise ValueError("nfe_noise must be positive to break identical IFE/NFE initialization")
        with torch.no_grad():
            for nfe in (self.nfe_rgb, self.nfe_thermal):
                for module in nfe.modules():
                    if isinstance(module, nn.Conv2d):
                        scale = module.weight.std().clamp_min(1e-4)
                        module.weight.add_(torch.randn_like(module.weight) * scale * nfe_noise)
        self.nc = len(class_names)
        self.names = dict(enumerate(class_names))
        self.yaml = copy.deepcopy(detector.yaml)
        self.yaml["nc"] = self.nc
        self.stride = self.ifd.stride
        self.gr = 1.0
        self.hyp: dict[str, float] = {}
        self.pretrained_source = source

    @property
    def model(self):
        """Compatibility surface required by YOLOv5 ComputeLoss."""
        return (*self.bfe.layers, *self.ife.layers, self.ifd)

    @classmethod
    def from_pretrained(cls, weights: str | Path, class_names: list[str],
                        nfe_noise: float = 0.05):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        # Official YOLOv5 checkpoints contain a serialized model object. Only
        # use a checkpoint from a trusted source; weights_only=True cannot load it.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source_model = (checkpoint.get("ema") or checkpoint.get("model"))
        if source_model is None or not hasattr(source_model, "yaml"):
            raise ValueError("Not an official YOLOv5 training checkpoint")
        source_model = source_model.float()
        yaml = copy.deepcopy(source_model.yaml)
        original_nc = int(yaml["nc"])
        detector = DetectionModel(yaml, ch=3, nc=len(class_names))
        source_state = source_model.state_dict()
        target_state = detector.state_dict()
        transferred = {
            key: value.float()
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        detector.load_state_dict(transferred, strict=False)
        detector.names = dict(enumerate(class_names))
        source = {
            "path": str(path),
            "original_classes": original_nc,
            "transferred_tensors": len(transferred),
            "target_tensors": len(target_state),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_date": str(checkpoint.get("date", "")),
        }
        return cls(detector, class_names, source, nfe_noise)

    def _apply(self, fn):
        super()._apply(fn)
        # YOLOv5 Detect keeps these tensors outside registered buffers.
        self.ifd.stride = fn(self.ifd.stride)
        self.ifd.grid = list(map(fn, self.ifd.grid))
        if isinstance(self.ifd.anchor_grid, list):
            self.ifd.anchor_grid = list(map(fn, self.ifd.anchor_grid))
        self.stride = self.ifd.stride
        return self

    @staticmethod
    def _slice_cache(cache: dict[int, Tensor], start: int, end: int):
        return {index: value[start:end] for index, value in cache.items()}

    def _features(self, images: Tensor):
        basic, _, basic_cache = self.bfe(ensure_three_channels(images))
        _, intrinsic, _ = self.ife(basic, basic_cache)
        return intrinsic

    def forward(self, images: Tensor):
        return self.ifd(list(self._features(images)))

    def forward_pair(self, rgb: Tensor, thermal: Tensor) -> dict[str, object]:
        rgb, thermal = ensure_three_channels(rgb), ensure_three_channels(thermal)
        if rgb.shape != thermal.shape:
            raise ValueError("Paired RGB and thermal tensors must have identical shapes")
        batch = rgb.shape[0]
        basic, _, basic_cache = self.bfe(torch.cat((rgb, thermal), dim=0))
        _, intrinsic, _ = self.ife(basic, basic_cache)
        predictions = self.ifd(list(intrinsic))
        if not self.training:
            decoded, raw = predictions
            return {
                "decoded_rgb": decoded[:batch],
                "decoded_thermal": decoded[batch:],
                "raw_rgb": tuple(value[:batch] for value in raw),
                "raw_thermal": tuple(value[batch:] for value in raw),
            }

        basic_rgb, basic_thermal = basic[:batch], basic[batch:]
        cache_rgb = self._slice_cache(basic_cache, 0, batch)
        cache_thermal = self._slice_cache(basic_cache, batch, 2 * batch)
        _, non_rgb, _ = self.nfe_rgb(basic_rgb, cache_rgb)
        _, non_thermal, _ = self.nfe_thermal(basic_thermal, cache_thermal)
        return {
            "pred_rgb": tuple(value[:batch] for value in predictions),
            "pred_thermal": tuple(value[batch:] for value in predictions),
            "intrinsic_rgb": tuple(value[:batch] for value in intrinsic),
            "intrinsic_thermal": tuple(value[batch:] for value in intrinsic),
            "non_rgb": non_rgb,
            "non_thermal": non_thermal,
        }


def targets_to_yolov5(targets: list[dict[str, Tensor]], image_hw: tuple[int, int]) -> Tensor:
    """Convert resized pixel xyxy targets to [image,class,cx,cy,w,h]."""
    height, width = image_hw
    rows = []
    for image_index, target in enumerate(targets):
        boxes = target["boxes"].float()
        labels = target["labels"].float()
        if not len(boxes):
            continue
        xywh = torch.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / (2 * width)
        xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / (2 * height)
        xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
        xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
        indices = torch.full((len(boxes), 1), float(image_index), device=boxes.device)
        rows.append(torch.cat((indices, labels[:, None], xywh), dim=1))
    if rows:
        return torch.cat(rows, dim=0)
    device = targets[0]["boxes"].device if targets else torch.device("cpu")
    return torch.empty((0, 6), device=device)


class OfficialYOLOv5UODLoss:
    def __init__(self, model: OfficialYOLOv5UOD, alpha: float = 1.0,
                 margin: float = 1.0, reduction: str = "pixel",
                 mask_padding: bool = False):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.detection = ComputeLoss(model)
        self.separation = FeatureSeparationLoss(margin, reduction, mask_padding)

    def __call__(self, outputs: dict[str, object], targets: list[dict[str, Tensor]],
                 image_hw: tuple[int, int], valid_mask: Tensor | None = None):
        yolo_targets = targets_to_yolov5(targets, image_hw)
        det_rgb, items_rgb = self.detection(outputs["pred_rgb"], yolo_targets)
        det_thermal, items_thermal = self.detection(outputs["pred_thermal"], yolo_targets)
        separation, separation_stats = self.separation(
            outputs["intrinsic_rgb"], outputs["intrinsic_thermal"],
            outputs["non_rgb"], outputs["non_thermal"], valid_mask)
        batch = max(1, len(targets))
        # ComputeLoss returns a batch sum. Scale the feature mean by the same
        # batch size so gradient accumulation exactly emulates a direct batch.
        loss = det_rgb + det_thermal + self.alpha * separation * batch
        items = items_rgb + items_thermal
        stats = {
            "loss": (loss / batch).detach(),
            "det_sum": ((det_rgb + det_thermal) / batch).detach(),
            "box": items[0],
            "obj": items[1],
            "cls": items[2],
            **separation_stats,
        }
        return loss, stats


__all__ = [
    "ModelEMA", "OfficialYOLOv5UOD", "OfficialYOLOv5UODLoss",
    "non_max_suppression", "smart_optimizer", "targets_to_yolov5",
]

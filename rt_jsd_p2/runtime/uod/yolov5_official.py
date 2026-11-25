"""UOD built from the pinned, official YOLOv5 v7.0 model and loss.

The YOLOv5 detector is split at its natural boundaries:
backbone (BFE, layers 0..9), neck (IFE, layers 10..23), and Detect (IFD,
layer 24).  The two NFE branches are independent copies of the pretrained
neck. With separate_ife=True, RGB and thermal also have independent IFE
necks, and all BFE routes are split before entering either neck.
This module intentionally lives beside the smaller reference model so
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
from .js_orth_feature import JSOrthFeatureLoss
from .js_alignment import JSDAlignmentLoss
from .kl_feature import SymmetricKLFeatureLoss
from .js_orth_modality import JSOrthModalityFeatureLoss
from .task_feature import TaskConditionedFeatureLoss


YOLOV5_ROOT = Path(__file__).resolve().parents[3] / "rt_jsd" / "runtime" / "third_party" / "yolov5"
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
    """Shared BFE/IFD, optional independent IFE necks, and training-only NFE."""

    def __init__(self, detector: DetectionModel, class_names: list[str],
                 source: dict[str, object], nfe_noise: float = 0.05,
                 use_nfe: bool = True, feature_objective: str = "hinge",
                 task_loss: dict | None = None,
                 modality_loss: dict | None = None, separate_ife: bool = False):
        super().__init__()
        layers = list(detector.model.children())
        if len(layers) != 25 or not isinstance(layers[-1], Detect):
            raise ValueError("Expected the 25-layer YOLOv5 v6/v7 detection topology")
        if [int(m.i) for m in layers] != list(range(25)):
            raise ValueError("Unexpected YOLOv5 layer indices")
        self.bfe = YOLOv5Stage(layers[:10], (4, 6))
        neck = YOLOv5Stage(layers[10:24], (17, 20, 23))
        self.separate_ife = bool(separate_ife)
        self.ife = None if self.separate_ife else neck
        self.ife_rgb = neck if self.separate_ife else None
        self.ife_thermal = copy.deepcopy(neck) if self.separate_ife else None
        self.ifd = layers[24]
        self.use_nfe = bool(use_nfe)
        if feature_objective not in {"hinge", "task_conditioned", "symmetric_kl", "js_orth_modality", "js_orth", "js_alignment"}:
            raise ValueError("Unknown feature_objective")
        if feature_objective in {"task_conditioned", "symmetric_kl", "js_orth_modality", "js_orth"} and not self.use_nfe:
            raise ValueError(f"{feature_objective} feature loss requires NFE")
        if feature_objective == "js_alignment" and self.use_nfe:
            raise ValueError("js_alignment uses IFE only; set use_nfe=False")
        self.feature_objective = feature_objective
        self.nfe_rgb: YOLOv5Stage | None = None
        self.nfe_thermal: YOLOv5Stage | None = None
        if self.use_nfe:
            self.nfe_rgb = copy.deepcopy(self.ife_rgb if self.separate_ife else self.ife)
            self.nfe_thermal = copy.deepcopy(self.ife_thermal if self.separate_ife else self.ife)
            if nfe_noise <= 0:
                raise ValueError("nfe_noise must be positive to break identical IFE/NFE initialization")
            with torch.no_grad():
                for nfe in (self.nfe_rgb, self.nfe_thermal):
                    for module in nfe.modules():
                        if isinstance(module, nn.Conv2d):
                            scale = module.weight.std().clamp_min(1e-4)
                            module.weight.add_(torch.randn_like(module.weight) * scale * nfe_noise)
        self.task_features = (
            TaskConditionedFeatureLoss(
                [layer.in_channels for layer in self.ifd.m], len(class_names),
                **(task_loss or {}))
            if feature_objective == "task_conditioned" else None
        )
        self.modality_features = (
            JSOrthModalityFeatureLoss(
                [layer.in_channels for layer in self.ifd.m], len(class_names),
                **(modality_loss or {}))
            if feature_objective == "js_orth_modality" else None
        )
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
        neck = self.ife_rgb if self.separate_ife else self.ife
        return (*self.bfe.layers, *neck.layers, self.ifd)

    @classmethod
    def from_pretrained(cls, weights: str | Path, class_names: list[str],
                        nfe_noise: float = 0.05, use_nfe: bool = True,
                        feature_objective: str = "hinge",
                        task_loss: dict | None = None,
                        modality_loss: dict | None = None, separate_ife: bool = False):
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
            "nfe_enabled": bool(use_nfe),
            "separate_ife": bool(separate_ife),
        }
        return cls(detector, class_names, source, nfe_noise, use_nfe,
                   feature_objective, task_loss, modality_loss, separate_ife)

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

    def _features(self, images: Tensor, modality: str | None = None):
        if modality not in (None, "rgb", "thermal"):
            raise ValueError("modality must be 'rgb' or 'thermal'")
        if self.separate_ife and modality is None:
            raise ValueError("Separate IFE inference requires modality='rgb' or 'thermal'")
        if self.separate_ife:
            neck = self.ife_rgb if modality == "rgb" else self.ife_thermal
        else:
            neck = self.ife
        basic, _, basic_cache = self.bfe(ensure_three_channels(images))
        _, intrinsic, _ = neck(basic, basic_cache)
        return intrinsic

    def forward(self, images: Tensor, modality: str | None = None):
        return self.ifd(list(self._features(images, modality)))

    def forward_pair(self, rgb: Tensor, thermal: Tensor, *,
                     return_features: bool = False) -> dict[str, object]:
        rgb, thermal = ensure_three_channels(rgb), ensure_three_channels(thermal)
        if rgb.shape != thermal.shape:
            raise ValueError("Paired RGB and thermal tensors must have identical shapes")
        batch = rgb.shape[0]
        basic, _, basic_cache = self.bfe(torch.cat((rgb, thermal), dim=0))
        # RGB [0:B], thermal [B:2B], including every routed BFE tensor.
        basic_rgb, basic_thermal = basic[:batch], basic[batch:]
        cache_rgb = self._slice_cache(basic_cache, 0, batch)
        cache_thermal = self._slice_cache(basic_cache, batch, 2 * batch)
        if self.separate_ife:
            _, intrinsic_rgb, _ = self.ife_rgb(basic_rgb, cache_rgb)
            _, intrinsic_thermal, _ = self.ife_thermal(basic_thermal, cache_thermal)
            intrinsic = tuple(torch.cat((rgb_level, thermal_level), dim=0)
                              for rgb_level, thermal_level in
                              zip(intrinsic_rgb, intrinsic_thermal, strict=True))
        else:
            _, intrinsic, _ = self.ife(basic, basic_cache)
        features = ({
            "intrinsic_rgb": tuple(value[:batch] for value in intrinsic),
            "intrinsic_thermal": tuple(value[batch:] for value in intrinsic),
        } if return_features or self.feature_objective == "js_alignment" else {})
        predictions = self.ifd(list(intrinsic))
        if not self.training:
            decoded, raw = predictions
            return {
                "decoded_rgb": decoded[:batch],
                "decoded_thermal": decoded[batch:],
                "raw_rgb": tuple(value[:batch] for value in raw),
                "raw_thermal": tuple(value[batch:] for value in raw),
                **features,
            }

        if not self.use_nfe:
            return {
                "pred_rgb": tuple(value[:batch] for value in predictions),
                "pred_thermal": tuple(value[batch:] for value in predictions),
                **features,
            }

        _, non_rgb, _ = self.nfe_rgb(basic_rgb, cache_rgb)
        _, non_thermal, _ = self.nfe_thermal(basic_thermal, cache_thermal)
        result = {
            "pred_rgb": tuple(value[:batch] for value in predictions),
            "pred_thermal": tuple(value[batch:] for value in predictions),
            "intrinsic_rgb": tuple(value[:batch] for value in intrinsic),
            "intrinsic_thermal": tuple(value[batch:] for value in intrinsic),
            "non_rgb": non_rgb,
            "non_thermal": non_thermal,
        }
        if self.task_features is not None:
            basic_pyramid = (basic_cache[4], basic_cache[6], basic)
            result["basic_rgb"] = tuple(value[:batch] for value in basic_pyramid)
            result["basic_thermal"] = tuple(value[batch:] for value in basic_pyramid)
        return result


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
                 mask_padding: bool = False, negative_mode: str = "hinge",
                 temperature: float = .2, kl_loss: dict | None = None,
                 js_orth_loss: dict | None = None,
                 js_alignment_loss: dict | None = None):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha
        self.feature_objective = model.feature_objective
        self.detection = ComputeLoss(model)
        if self.feature_objective == "js_alignment":
            self.separation = JSDAlignmentLoss(**(js_alignment_loss or {}))
        elif not model.use_nfe:
            self.separation = None
        elif self.feature_objective == "task_conditioned":
            self.separation = model.task_features
        elif self.feature_objective == "js_orth_modality":
            self.separation = model.modality_features
        elif self.feature_objective == "symmetric_kl":
            self.separation = SymmetricKLFeatureLoss(**(kl_loss or {}))
        elif self.feature_objective == "js_orth":
            self.separation = JSOrthFeatureLoss(**(js_orth_loss or {}))
        else:
            self.separation = FeatureSeparationLoss(
                margin, reduction, mask_padding, negative_mode, temperature)

    def __call__(self, outputs: dict[str, object], targets: list[dict[str, Tensor]],
                 image_hw: tuple[int, int], valid_mask: Tensor | None = None):
        yolo_targets = targets_to_yolov5(targets, image_hw)
        det_rgb, items_rgb = self.detection(outputs["pred_rgb"], yolo_targets)
        det_thermal, items_thermal = self.detection(outputs["pred_thermal"], yolo_targets)
        batch = max(1, len(targets))
        if self.separation is None:
            separation = (det_rgb + det_thermal).new_zeros(())
            separation_stats = {"fs": separation.detach()}
        else:
            if self.feature_objective == "js_alignment":
                separation, separation_stats = self.separation(
                    outputs["intrinsic_rgb"], outputs["intrinsic_thermal"], valid_mask,
                    targets=targets, image_hw=image_hw)
            elif self.feature_objective == "task_conditioned":
                separation, separation_stats = self.separation(
                    outputs, targets, image_hw, valid_mask)
            elif self.feature_objective == "js_orth_modality":
                _, _, matched_indices, _ = self.detection.build_targets(
                    outputs["pred_rgb"], yolo_targets)
                separation, separation_stats = self.separation(
                    outputs, matched_indices, valid_mask)
            else:
                separation, separation_stats = self.separation(
                    outputs["intrinsic_rgb"], outputs["intrinsic_thermal"],
                    outputs["non_rgb"], outputs["non_thermal"], valid_mask)
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

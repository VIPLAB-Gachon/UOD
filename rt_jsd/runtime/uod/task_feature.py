"""Task conditioned feature separation for the YOLOv5 UOD experiment.

The dataset's detection labels define what is relevant.  A shared semantic
probe supervises IFE and aligns paired RGB/thermal predictions.  An
adversarial probe discourages target information in NFE, while sampled
non-object reconstruction encourages NFE to preserve task-irrelevant content.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: Tensor) -> Tensor:
        return -gradient


def reverse_gradient(value: Tensor) -> Tensor:
    return _ReverseGradient.apply(value)


class TaskConditionedFeatureLoss(nn.Module):
    """Learn from matched object centers and balanced non-object locations.

    Class zero in the auxiliary probes means background; detector class k is
    represented by k+1.  This also gives a useful task when nc=1 (LLVIP).
    """

    def __init__(self, channels: list[int], num_classes: int, *,
                 max_background: int = 16, temperature: float = 1.0,
                 semantic_weight: float = 0.05, align_weight: float = 0.1,
                 orth_weight: float = 0.05, adversarial_weight: float = 0.02,
                 reconstruction_weight: float = 0.05):
        super().__init__()
        if len(channels) != 3 or num_classes < 1 or max_background < 1 or temperature <= 0:
            raise ValueError("Invalid task conditioned feature configuration")
        weights = (semantic_weight, align_weight, orth_weight,
                   adversarial_weight, reconstruction_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("Feature loss weights must be non-negative")
        self.num_classes = num_classes
        self.max_background = max_background
        self.temperature = temperature
        self.weights = weights
        # The final YOLOv5 neck outputs and BFE pyramid have matching channels.
        self.semantic = nn.ModuleList(nn.Linear(c, num_classes + 1) for c in channels)
        self.nfe_probe = nn.ModuleList(
            nn.Sequential(nn.Linear(c, min(c, 128)), nn.SiLU(),
                          nn.Linear(min(c, 128), num_classes + 1)) for c in channels)
        self.reconstructor = nn.ModuleList(
            nn.Linear(c, c) for c in channels)

    @staticmethod
    def _tokens(feature: Tensor, batch_index: int, ys: Tensor, xs: Tensor) -> Tensor:
        return feature[batch_index, :, ys, xs].transpose(0, 1).float()

    @torch.no_grad()
    def _locations(self, target: dict[str, Tensor], height: int, width: int,
                   image_hw: tuple[int, int], valid: Tensor | None,
                   device: torch.device) -> tuple[Tensor, Tensor, Tensor, int]:
        image_h, image_w = image_hw
        boxes = target["boxes"].to(device)
        labels = target["labels"].to(device).long()
        if labels.numel() and ((labels < 0).any() or (labels >= self.num_classes).any()):
            raise ValueError("Target class outside auxiliary probe range")

        cx = ((boxes[:, 0] + boxes[:, 2]) * width / (2 * image_w)).long().clamp(0, width - 1)
        cy = ((boxes[:, 1] + boxes[:, 3]) * height / (2 * image_h)).long().clamp(0, height - 1)
        if valid is not None and len(cx):
            keep = valid[cy, cx]
            cx, cy, labels = cx[keep], cy[keep], labels[keep]
        positive_count = len(cx)

        candidates = torch.randperm(height * width, device=device)[:self.max_background * 8]
        ys, xs = candidates // width, candidates % width
        if valid is not None:
            keep = valid[ys, xs]
            ys, xs = ys[keep], xs[keep]
        if len(boxes) and len(xs):
            px = (xs.float() + 0.5) * image_w / width
            py = (ys.float() + 0.5) * image_h / height
            inside = ((px[:, None] >= boxes[None, :, 0]) &
                      (px[:, None] <= boxes[None, :, 2]) &
                      (py[:, None] >= boxes[None, :, 1]) &
                      (py[:, None] <= boxes[None, :, 3])).any(dim=1)
            ys, xs = ys[~inside], xs[~inside]
        background_count = min(self.max_background, max(positive_count, 1), len(xs))
        ys, xs = ys[:background_count], xs[:background_count]
        all_y = torch.cat((cy, ys))
        all_x = torch.cat((cx, xs))
        classes = torch.cat((labels + 1, torch.zeros(background_count, device=device, dtype=torch.long)))
        return all_y, all_x, classes, positive_count

    def forward(self, outputs: dict[str, object], targets: list[dict[str, Tensor]],
                image_hw: tuple[int, int], valid_mask: Tensor | None = None):
        names = ("semantic", "align_js", "orth", "adversarial", "reconstruction")
        totals = {name: [] for name in names}
        pyramids = (outputs["intrinsic_rgb"], outputs["intrinsic_thermal"],
                    outputs["non_rgb"], outputs["non_thermal"],
                    outputs["basic_rgb"], outputs["basic_thermal"])
        if any(len(pyramid) != 3 for pyramid in pyramids):
            raise ValueError("Expected three-level BFE, IFE and NFE pyramids")
        if any(pyramid[0].shape[0] != len(targets) for pyramid in pyramids):
            raise ValueError("Feature batch and target batch disagree")

        for level in range(3):
            ir, it, nr, nt, br, bt = (p[level] for p in pyramids)
            if len({tuple(t.shape) for t in (ir, it, nr, nt, br, bt)}) != 1:
                raise ValueError("IFE, NFE and BFE level shapes must match")
            height, width = ir.shape[-2:]
            valid = None
            if valid_mask is not None:
                valid = F.interpolate(valid_mask.float(), size=(height, width),
                                      mode="nearest")[:, 0].bool()
            for batch_index, target in enumerate(targets):
                ys, xs, classes, positive_count = self._locations(
                    target, height, width, image_hw,
                    valid[batch_index] if valid is not None else None, ir.device)
                if not len(classes):
                    continue
                i_r = self._tokens(ir, batch_index, ys, xs)
                i_t = self._tokens(it, batch_index, ys, xs)
                n_r = self._tokens(nr, batch_index, ys, xs)
                n_t = self._tokens(nt, batch_index, ys, xs)
                b_r = self._tokens(br, batch_index, ys, xs).detach()
                b_t = self._tokens(bt, batch_index, ys, xs).detach()

                logits_r = self.semantic[level](i_r)
                logits_t = self.semantic[level](i_t)
                totals["semantic"].append(
                    (F.cross_entropy(logits_r, classes) +
                     F.cross_entropy(logits_t, classes)) * 0.5)
                if positive_count:
                    p = F.softmax(logits_r[:positive_count] / self.temperature, dim=-1)
                    q = F.softmax(logits_t[:positive_count] / self.temperature, dim=-1)
                    middle = (p + q) * 0.5
                    js = (F.kl_div(middle.clamp_min(1e-8).log(), p, reduction="batchmean") +
                          F.kl_div(middle.clamp_min(1e-8).log(), q, reduction="batchmean")) * 0.5
                    totals["align_js"].append(js)

                # Keep IFE detection features fixed for the orthogonality term.
                orth = (F.cosine_similarity(i_r.detach(), n_r, dim=-1).square().mean() +
                        F.cosine_similarity(i_t.detach(), n_t, dim=-1).square().mean()) * 0.5
                totals["orth"].append(orth)

                adv_r = self.nfe_probe[level](reverse_gradient(n_r))
                adv_t = self.nfe_probe[level](reverse_gradient(n_t))
                totals["adversarial"].append(
                    (F.cross_entropy(adv_r, classes) +
                     F.cross_entropy(adv_t, classes)) * 0.5)

                # The decoder sees only NFE at known non-target locations, so
                # it cannot ignore NFE and reconstruct from IFE alone.
                if positive_count < len(classes):
                    n_r_bg, n_t_bg = n_r[positive_count:], n_t[positive_count:]
                    b_r_bg, b_t_bg = b_r[positive_count:], b_t[positive_count:]
                    reconstructed_r = self.reconstructor[level](n_r_bg)
                    reconstructed_t = self.reconstructor[level](n_t_bg)
                    rec = (F.smooth_l1_loss(
                        F.layer_norm(reconstructed_r, (b_r.shape[-1],)),
                        F.layer_norm(b_r_bg, (b_r.shape[-1],))) +
                        F.smooth_l1_loss(
                        F.layer_norm(reconstructed_t, (b_t.shape[-1],)),
                        F.layer_norm(b_t_bg, (b_t.shape[-1],)))) * 0.5
                    totals["reconstruction"].append(rec)

        zero = pyramids[0][0].float().sum() * 0.0
        values = {name: torch.stack(items).mean() if items else zero
                  for name, items in totals.items()}
        loss = sum(weight * values[name] for weight, name in zip(self.weights, names))
        stats = {name: value.detach() for name, value in values.items()}
        stats["fs"] = loss.detach()
        return loss, stats

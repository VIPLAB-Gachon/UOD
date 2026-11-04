"""GT-supervised distribution detection and object-selective cross-modal JSD."""
from __future__ import annotations
import torch
from torch import Tensor
from .localization_distribution import (decode_distribution, distribution_focal_loss,
    distribution_focal_loss_per_edge, directed_jsd, directed_jsd_per_edge)
from utils.loss import ComputeLoss
from utils.metrics import bbox_iou


EDGE_GATE_STATS = (
    'eligible_edges_per_pair', 'selected_edges_per_pair', 'blocked_edges_per_pair',
    'selected_rgb_teacher_edges_per_pair', 'selected_thermal_teacher_edges_per_pair',
    'edge_gate_blocked_fraction',
)


def gt_better_edge_mask(student_logits: Tensor, teacher_logits: Tensor,
                        target_distances: Tensor, support: Tensor,
                        margin: float = 1e-6) -> Tensor:
    """Keep an edge only when the object teacher is better by both GT tests.

    Compare signed-distance expectations in the same anchor units, and the
    same two-bin DFL used for GT supervision. Selection never carries gradients.
    """
    if margin < 0:
        raise ValueError('Edge comparison margin must be nonnegative')
    with torch.no_grad():
        student = student_logits.detach().float()
        teacher = teacher_logits.detach().float()
        target = target_distances.detach().float()
        support = support.detach().to(device=student.device, dtype=torch.float32)
        student_distance = (student.softmax(-1) * support).sum(-1)
        teacher_distance = (teacher.softmax(-1) * support).sum(-1)
        student_error = (student_distance - target).abs()
        teacher_error = (teacher_distance - target).abs()
        student_dfl = distribution_focal_loss_per_edge(student, target, support)
        teacher_dfl = distribution_focal_loss_per_edge(teacher, target, support)
        return ((teacher_error + margin < student_error)
                & (teacher_dfl + margin < student_dfl))


def bidirectional_gt_better_edge_masks(rgb_logits: Tensor, thermal_logits: Tensor,
                                      target_distances: Tensor, support: Tensor,
                                      margin: float = 1e-6):
    """Return detached, mutually exclusive RGB/thermal teacher masks per edge.

    The caller retains the existing common-candidate selection and object gate.
    A direction requires agreement of GT distance error and GT DFL winners.
    """
    rgb_teacher = gt_better_edge_mask(
        thermal_logits, rgb_logits, target_distances, support, margin)
    thermal_teacher = gt_better_edge_mask(
        rgb_logits, thermal_logits, target_distances, support, margin)
    return rgb_teacher, thermal_teacher


def select_object_pairs(keys: Tensor, gt_ids: Tensor, quality_rgb: Tensor,
                        quality_thermal: Tensor, num_gt: int,
                        min_iou: float, min_gap: float):
    """One shared candidate per GT; exclude keys assigned to several GTs.

    Returns original candidate row indices, teacher_is_rgb, and diagnostics.
    All decisions are detached. Lexicographic candidate key resolves ties.
    """
    with torch.no_grad():
        if not 0 <= min_iou <= 1 or not 0 <= min_gap <= 1:
            raise ValueError('Teacher IoU/gap thresholds must be in [0,1]')
        empty = keys.new_empty(0)
        diag = {'gt_total': num_gt, 'gt_with_candidate': 0, 'gt_missing': num_gt,
                'candidate_ambiguous': 0, 'selected_rgb_teacher': 0,
                'selected_thermal_teacher': 0, 'low_quality': 0, 'small_gap': 0}
        if not len(keys):
            return empty, empty.bool(), diag
        if keys.dtype != torch.long or gt_ids.dtype != torch.long:
            raise TypeError('Candidate keys and GT IDs must be int64')
        if int(gt_ids.min()) < 0 or int(gt_ids.max()) >= num_gt:
            raise ValueError('Candidate GT ID outside target rows')
        # Sort by key then GT ID, and remove duplicate assignment rows.
        order = torch.argsort(keys * (num_gt + 1) + gt_ids, stable=True)
        k, g = keys[order], gt_ids[order]
        unique_pair = torch.ones_like(k, dtype=torch.bool)
        unique_pair[1:] = (k[1:] != k[:-1]) | (g[1:] != g[:-1])
        order, k, g = order[unique_pair], k[unique_pair], g[unique_pair]
        _, counts = torch.unique_consecutive(k, return_counts=True)
        unambiguous = torch.repeat_interleave(counts, counts) == 1
        diag['candidate_ambiguous'] = int((counts > 1).sum())
        order, g = order[unambiguous], g[unambiguous]
        selected, directions = [], []
        for gt_id in range(num_gt):
            rows = order[g == gt_id]
            if not len(rows):
                continue
            diag['gt_with_candidate'] += 1
            qr, qt = quality_rgb[rows].detach(), quality_thermal[rows].detach()
            qmax = torch.maximum(qr, qt)
            # rows already sorted by ascending candidate key; first tie wins.
            local = int(qmax.argmax())
            r, t = float(qr[local]), float(qt[local])
            if max(r, t) < min_iou:
                diag['low_quality'] += 1
            elif abs(r - t) < min_gap or r == t:
                diag['small_gap'] += 1
            else:
                selected.append(rows[local])
                directions.append(r > t)
                diag['selected_rgb_teacher' if r > t else 'selected_thermal_teacher'] += 1
        diag['gt_missing'] = num_gt - diag['gt_with_candidate']
        if not selected:
            return empty, empty.bool(), diag
        return torch.stack(selected), torch.tensor(directions, device=keys.device, dtype=torch.bool), diag


class DistributionDetectionLoss(ComputeLoss):
    """YOLOv5 assignment/objectness/classification with DFL + decoded CIoU."""
    def __init__(self, model, dfl_weight=.01):
        super().__init__(model)
        if dfl_weight < 0:
            raise ValueError('DFL weight must be non-negative')
        self.head = model.ifd
        self.dfl_weight = float(dfl_weight)

    def loss_and_candidates(self, predictions, targets):
        # build_targets uses class only as a carried label. Carry the original
        # target row ID there, then recover its true class for classification.
        assignment_targets = targets.clone()
        assignment_targets[:, 1] = torch.arange(len(targets), device=targets.device)
        ids, tbox, indices, anchors = self.build_targets(predictions, assignment_targets)
        zero = predictions[0].new_zeros((), dtype=torch.float32)
        lbox, lobj, lcls, ldfl = zero, zero, zero, zero
        candidates = {'keys': [], 'ids': [], 'batch_ids': [], 'logits': [], 'quality': [], 'target_distances': []}
        invalid, matched, low, high = 0, 0, float('inf'), -float('inf')
        offset = 0
        for level, raw in enumerate(predictions):
            b, a, gy, gx = indices[level]
            tobj = torch.zeros(raw.shape[:4], device=raw.device, dtype=raw.dtype)
            count = len(b)
            if count:
                chosen = raw[b, a, gy, gx]
                logits = chosen[:, :4 * self.head.bins].reshape(-1, 4, self.head.bins)
                grid = torch.stack((gx, gy), dim=1).float()
                points = grid + .5
                pred_box, valid = decode_distribution(logits, points, anchors[level], self.head.support)
                target_box = tbox[level].float().clone()
                target_box[:, :2] += grid
                half = target_box[:, 2:] / 2
                left_top = (points - (target_box[:, :2] - half)) / anchors[level]
                right_bottom = ((target_box[:, :2] + half) - points) / anchors[level]
                distances = torch.cat((left_top, right_bottom), dim=1)
                # The DFL helper refuses unsupported targets; never silently clip GT.
                ldfl = ldfl + distribution_focal_loss(logits, distances, self.head.support).mean()
                ciou = bbox_iou(pred_box, target_box, CIoU=True).reshape(-1)
                lbox = lbox + (1.0 - ciou).mean()
                quality = bbox_iou(pred_box.detach(), target_box, CIoU=False).reshape(-1).clamp(0, 1)
                object_quality = ciou.detach().clamp(0).to(tobj.dtype)
                tobj[b, a, gy, gx] = object_quality
                if self.nc > 1:
                    pcls = chosen[:, 4 * self.head.bins + 1:]
                    cls_target = torch.full_like(pcls, self.cn)
                    classes = targets[ids[level], 1].long()
                    cls_target[torch.arange(count, device=b.device), classes] = self.cp
                    lcls = lcls + self.BCEcls(pcls, cls_target)
                key = offset + ((b * self.na + a) * raw.shape[2] + gy) * raw.shape[3] + gx
                for name, value in [('keys', key), ('ids', ids[level]), ('batch_ids', b),
                                    ('logits', logits), ('quality', quality), ('target_distances', distances.detach())]:
                    candidates[name].append(value)
                invalid += int((~valid).sum())
                matched += count
                low = min(low, float(distances.detach().min()))
                high = max(high, float(distances.detach().max()))
            lobj = lobj + self.BCEobj(raw[..., 4 * self.head.bins], tobj) * self.balance[level]
            offset += raw.shape[0] * self.na * raw.shape[2] * raw.shape[3]
        lbox = lbox * self.hyp['box']
        lobj = lobj * self.hyp['obj']
        lcls = lcls * self.hyp['cls']
        ldfl = ldfl * self.dfl_weight
        items = torch.stack((lbox, lobj, lcls, ldfl))
        batch = predictions[0].shape[0]
        empty_logits = predictions[0].new_empty((0, 4, self.head.bins))
        for name, values in candidates.items():
            candidates[name] = torch.cat(values) if values else (
                empty_logits if name == 'logits' else zero.new_empty((0, 4)) if name == 'target_distances'
                else zero.new_empty(0) if name == 'quality'
                else torch.empty(0, device=zero.device, dtype=torch.long))
        return items.sum() * batch, items.detach(), candidates, {
            'matched_candidates': matched, 'invalid_candidates': invalid,
            'target_support_min': low if matched else 0.,
            'target_support_max': high if matched else 0.}


class DistributionUODLoss:
    """Both GT losses plus gated JSD with detached object or edge teachers."""
    def __init__(self, model, alpha=0., dfl_weight=.01, min_teacher_iou=.7,
                 min_iou_gap=.1, temperature=1., edge_gate='off'):
        if alpha < 0 or temperature <= 0:
            raise ValueError('Invalid JSD coefficient or temperature')
        if not 0 <= min_teacher_iou <= 1 or not 0 <= min_iou_gap <= 1:
            raise ValueError('Invalid teacher thresholds')
        if edge_gate not in {'off', 'gt_better', 'gt_better_bidirectional'}:
            raise ValueError("Unsupported edge_gate")
        self.edge_gate = edge_gate
        self.alpha = float(alpha)
        self.separation = None  # No legacy IFE feature alignment.
        self.detection = DistributionDetectionLoss(model, dfl_weight=dfl_weight)
        self.min_teacher_iou = float(min_teacher_iou)
        self.min_iou_gap = float(min_iou_gap)
        self.temperature = float(temperature)

    def __call__(self, outputs, targets, image_hw, valid_mask=None):
        # Import here to avoid a circular dependency with the model wrapper.
        from .yolov5_official import targets_to_yolov5
        yolo_targets = targets_to_yolov5(targets, image_hw)
        dr, ir, cr, ar = self.detection.loss_and_candidates(outputs['pred_rgb'], yolo_targets)
        dt, it, ct, at = self.detection.loss_and_candidates(outputs['pred_thermal'], yolo_targets)
        if not torch.equal(cr['keys'], ct['keys']) or not torch.equal(cr['ids'], ct['ids']):
            raise RuntimeError('RGB and Thermal assignments are not in the same coordinates')
        rows, teacher_rgb, diag = select_object_pairs(
            cr['keys'], cr['ids'], cr['quality'], ct['quality'], len(yolo_targets),
            self.min_teacher_iou, self.min_iou_gap)
        batch = len(targets)
        if batch != outputs['pred_rgb'][0].shape[0] or batch == 0:
            raise ValueError('Target list must match the nonempty image batch')
        js = (dr + dt).new_zeros(())
        mean_teacher = mean_student = js.detach()
        teacher_sum = student_sum = js.detach()
        eligible_edges = selected_edges = rgb_edges = thermal_edges = 0
        reverse_edges = 0
        if len(rows):
            rgb, thermal = cr['logits'][rows], ct['logits'][rows]
            eligible_edges = 4 * len(rows)
            if self.edge_gate in {'gt_better', 'gt_better_bidirectional'}:
                if not torch.equal(cr['target_distances'][rows], ct['target_distances'][rows]):
                    raise RuntimeError('RGB and Thermal edge targets differ')
            if self.edge_gate == 'gt_better_bidirectional':
                rgb_teacher_mask, thermal_teacher_mask = bidirectional_gt_better_edge_masks(
                    rgb, thermal, cr['target_distances'][rows], self.detection.head.support)
                edge_mask = rgb_teacher_mask | thermal_teacher_mask
                edge_teacher_rgb = rgb_teacher_mask
                direction = edge_teacher_rgb[..., None]
                teacher = torch.where(direction, rgb, thermal).detach()
                student = torch.where(direction, thermal, rgb)
                losses = (directed_jsd_per_edge(student, teacher, self.temperature)
                          * edge_mask).sum(-1) / 4.0
                reverse_edges = int((edge_mask &
                    (edge_teacher_rgb != teacher_rgb[:, None])).sum())
            else:
                edge_teacher_rgb = teacher_rgb[:, None].expand(-1, 4)
                direction = teacher_rgb[:, None, None]
                teacher = torch.where(direction, rgb, thermal).detach()
                student = torch.where(direction, thermal, rgb)
            if self.edge_gate == 'gt_better':
                edge_mask = gt_better_edge_mask(
                    student, teacher, cr['target_distances'][rows], self.detection.head.support)
                # Retain the original four-edge normalization even when some
                # edges are rejected. Never amplify the remaining edges.
                losses = (directed_jsd_per_edge(student, teacher, self.temperature)
                          * edge_mask).sum(-1) / 4.0
            elif self.edge_gate == 'off':
                edge_mask = torch.ones((len(rows), 4), device=student.device, dtype=torch.bool)
                losses = directed_jsd(student, teacher, self.temperature)
            selected_edges = int(edge_mask.sum())
            rgb_edges = int((edge_mask & edge_teacher_rgb).sum())
            thermal_edges = selected_edges - rgb_edges
            selected_images = cr['batch_ids'][rows]
            # Sum per image, divide by all of its GTs (not selected GTs).
            # Explicit per-image sums avoid nondeterministic CUDA scatter_add.
            per_image = [losses[selected_images == i].sum() / max(1, len(t['boxes']))
                         for i, t in enumerate(targets)]
            js = torch.stack(per_image).mean()
            qr, qt = cr['quality'][rows], ct['quality'][rows]
            teacher_quality = torch.where(teacher_rgb, qr, qt).detach()
            student_quality = torch.where(teacher_rgb, qt, qr).detach()
            mean_teacher, mean_student = teacher_quality.mean(), student_quality.mean()
            teacher_sum, student_sum = teacher_quality.sum() / batch, student_quality.sum() / batch
        loss = dr + dt + batch * self.alpha * js
        items = ir + it
        # Per-image counts: the trainer batch-weights these statistics. Multiply
        # epoch means by its pair count to recover totals (apart from rounding).
        stats = {'loss': (loss / batch).detach(), 'det_sum': ((dr + dt) / batch).detach(),
                 'box': items[0], 'obj': items[1], 'cls': items[2], 'dfl': items[3],
                 'fs': js.detach(), 'localization_jsd': js.detach(),
                 'localization_jsd_weighted': (self.alpha * js).detach(),
                 'teacher_iou_selected': mean_teacher, 'student_iou_selected': mean_student,
                 'teacher_iou_sum_per_pair': teacher_sum, 'student_iou_sum_per_pair': student_sum,
                 'invalid_rgb_per_pair': loss.new_tensor(ar['invalid_candidates'] / batch),
                 'invalid_thermal_per_pair': loss.new_tensor(at['invalid_candidates'] / batch),
                 'matched_per_pair': loss.new_tensor(ar['matched_candidates'] / batch),
                 'target_support_min': loss.new_tensor(ar['target_support_min']),
                 'target_support_max': loss.new_tensor(ar['target_support_max'])}
        stats.update({name + '_per_pair': loss.new_tensor(value / batch) for name, value in diag.items()})
        stats.update({
            'eligible_edges_per_pair': loss.new_tensor(eligible_edges / batch),
            'selected_edges_per_pair': loss.new_tensor(selected_edges / batch),
            'blocked_edges_per_pair': loss.new_tensor((eligible_edges - selected_edges) / batch),
            'selected_rgb_teacher_edges_per_pair': loss.new_tensor(rgb_edges / batch),
            'selected_thermal_teacher_edges_per_pair': loss.new_tensor(thermal_edges / batch),
            'edge_gate_blocked_fraction': loss.new_tensor(
                (eligible_edges - selected_edges) / eligible_edges if eligible_edges else 0.),
        })
        if self.edge_gate == 'gt_better_bidirectional':
            stats.update({
                'reverse_edges_per_pair': loss.new_tensor(reverse_edges / batch),
                'object_direction_edges_per_pair': loss.new_tensor(
                    (selected_edges - reverse_edges) / batch),
            })
        return loss, stats

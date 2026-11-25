"""Original GT/JSD plus RGB within-GT confidence ranking for three classes.

No new parameters, assignment, objectness targets, or inference components.
Collision exclusion uses ALL GT links before the focus-class validity mask.
"""
import torch
from uod.localization_loss import DistributionDetectionLoss, DistributionUODLoss
from uod.yolov5_official import targets_to_yolov5
from uod.focus_ranking import within_gt_confidence_ranking


def scheduled_ranking_beta(config, progress):
    return float(config['beta']) * min(1., max(0.,
        (progress - float(config['start_epoch'])) / float(config['ramp_epochs'])))


class CaptureDetection(DistributionDetectionLoss):
    def loss_and_candidates(self, predictions, targets):
        result = super().loss_and_candidates(predictions, targets)
        self.captured.append(result[2])
        return result


class FocusObjective:
    def __init__(self, model, cfg):
        self.original = DistributionUODLoss(model, alpha=cfg['loss']['alpha'], **cfg['loss']['localization'])
        self.original.detection = CaptureDetection(model, cfg['loss']['localization']['dfl_weight'])
        self.class_ids = tuple(cfg['loss']['ranking']['class_ids'])
        self.beta = 0.
        self.capture_graph = False

    @property
    def alpha(self):
        return self.original.alpha

    @alpha.setter
    def alpha(self, value):
        self.original.alpha = value

    def __call__(self, outputs, targets, image_hw, valid_mask=None):
        detector = self.original.detection
        detector.captured = []
        original, stats = self.original(outputs, targets, image_hw, valid_mask)
        rgb, _ = detector.captured
        detector.captured = []
        yolo = targets_to_yolov5(targets, image_hw)
        batch = len(targets)
        if self.beta == 0 and not self.capture_graph:
            # Preserve the exact baseline arithmetic during the common warm-up.
            zero = original.detach().new_zeros(())
            rank = zero
            diag = {key: zero for key in diagnostic_keys(self.class_ids)}
            for class_id in self.class_ids:
                diag[f'class_{class_id}_gt_per_pair'] = (yolo[:, 1] == class_id).sum().float() / batch
        else:
            head = detector.head
            # Keys follow head-level offsets; each level is [B,anchor,y,x].
            logits = torch.cat([p[..., 4 * head.bins:].reshape(-1, 1 + head.nc)
                                for p in outputs['pred_rgb']])[rgb['keys']].float()
            classes = yolo[:, 1].long()
            assigned = classes[rgb['ids']]
            scores = logits[:, 0].sigmoid() * logits[:, 1:].sigmoid().gather(1, assigned[:, None]).squeeze(1)
            mean = (rgb['logits'].detach().float().softmax(-1) * head.support.float()).sum(-1)
            valid = ((mean[:, 0] + mean[:, 2] > 0) & (mean[:, 1] + mean[:, 3] > 0)
                     & torch.isfinite(mean).all(1)
                     & torch.isin(assigned, assigned.new_tensor(self.class_ids)))
            rank, diag = within_gt_confidence_ranking(scores, rgb['quality'], rgb['keys'],
                rgb['ids'], valid, yolo[:, 0].long(), batch, classes, self.class_ids)
        if self.capture_graph:
            self.last_rank_loss = rank
        loss = original if self.beta == 0 else original + batch * self.beta * rank
        return loss, dict(stats, loss=(loss / batch).detach(),
            original_loss=(original / batch).detach(), ranking=rank.detach(),
            ranking_weighted=(self.beta * rank).detach(), ranking_beta=rank.detach().new_tensor(self.beta),
            **{'ranking_' + key: value for key, value in diag.items()})


def diagnostic_keys(classes):
    return [name + '_per_pair' for name in ('candidates', 'ambiguous_candidates',
        'invalid_candidates', 'ordered_pairs', 'inverted_pairs', 'equal_score_pairs',
        'margin_violations', 'gt_with_pairs')] + [f'class_{c}_{key}' for c in classes
        for key in ('contribution', 'gt_per_pair', 'ordered_pairs_per_pair', 'gt_with_pairs_per_pair')]


def active_gpu_check(model, criterion, batch, cfg):
    """Disposable live-output check after DDP preflight; never updates weights."""
    model.train()
    criterion.capture_graph = True
    with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['training']['amp']):
        outputs = model.forward_pair(batch['rgb'], batch['thermal'])
        native = DistributionUODLoss(model, alpha=criterion.alpha, **cfg['loss']['localization'])
        expected, _ = native(outputs, batch['targets'], batch['rgb'].shape[-2:], batch['valid_mask'])
        criterion.beta = 0.
        control, _ = criterion(outputs, batch['targets'], batch['rgb'].shape[-2:], batch['valid_mask'])
        assert torch.equal(control, expected), 'beta0 must reproduce native detector/JSD exactly'
        criterion.beta = cfg['loss']['ranking']['beta']
        active, stats = criterion(outputs, batch['targets'], batch['rgb'].shape[-2:], batch['valid_mask'])
    rank = criterion.last_rank_loss
    assert torch.isfinite(active) and torch.isfinite(rank) and float(rank) > 0
    torch.testing.assert_close(active, expected + len(batch['targets']) * criterion.beta * rank, rtol=0, atol=0)
    rgb, thermal = outputs['pred_rgb'], outputs['pred_thermal']
    grads = torch.autograd.grad(rank, tuple(rgb) + tuple(thermal), allow_unused=True)
    offset = 4 * model.ifd.bins
    non_focus = [1 + c for c in range(model.ifd.nc) if c not in criterion.class_ids]
    assert all(g is None for g in grads[len(rgb):]), 'Ranking must not depend directly on Thermal outputs'
    rgb_norm = 0.
    for grad in grads[:len(rgb)]:
        if grad is None:
            continue
        assert torch.isfinite(grad).all()
        assert not torch.count_nonzero(grad[..., :offset]), 'IoU/localization target must be detached'
        assert not torch.count_nonzero(grad[..., offset:][..., non_focus]), 'Other class channels must have zero direct gradient'
        rgb_norm += float(grad.abs().sum())
    assert rgb_norm > 0
    criterion.capture_graph = False
    del criterion.last_rank_loss
    return dict(status='passed', beta_zero_native_loss_exact=True, additive_weight_exact=True,
        active_rank_loss=float(rank), rgb_output_gradient_l1=rgb_norm,
        thermal_output_gradient_absent=True, regression_output_gradient_zero=True,
        non_focus_class_output_gradient_zero=True,
        class_gt_counts={str(c): float(stats[f'ranking_class_{c}_gt_per_pair']) * len(batch['targets']) for c in criterion.class_ids},
        scope='Raw output gradients only; shared backbone/head changes can still affect all classes and modes during full training')

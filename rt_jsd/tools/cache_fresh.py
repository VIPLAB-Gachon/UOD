"""One fresh frozen-EMA pass for sparse Train and dense Validation candidates."""
from __future__ import annotations
import hashlib
import json

import numpy as np
import torch
from torchvision.ops import box_iou

from calibration_common import atomic_torch, now, sha, tensor_digest, write
from distribution_features import quality_features


def assignment_keys(raw_levels, targets, loss):
    carried = targets.clone()
    carried[:, 1] = torch.arange(len(targets), device=targets.device)
    gt_ids, _, indices, _ = loss.build_targets(raw_levels, carried)
    images, keys, ids, offset = [], [], [], 0
    for raw, gt, index in zip(raw_levels, gt_ids, indices):
        image, anchor, gy, gx = index
        _, na, ny, nx, _ = raw.shape
        images.append(image)
        keys.append(offset + (anchor * ny + gy) * nx + gx)
        ids.append(gt.long())
        offset += na * ny * nx
    return torch.cat(images), torch.cat(keys), torch.cat(ids), offset


def negative_keys(count, positive, image_id, mode, seed, maximum):
    available = np.ones(count, dtype=np.bool_)
    available[positive.detach().cpu().numpy()] = False
    remaining = np.flatnonzero(available)
    text = '%d|%s|%s' % (seed, image_id, mode)
    number = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')
    selected = np.sort(np.random.default_rng(number).choice(
        remaining, min(len(remaining), maximum), replace=False))
    return torch.from_numpy(selected).to(positive.device), len(remaining), hashlib.sha256(selected.tobytes()).hexdigest()


@torch.no_grad()
def assigned_targets(boxes, gt_boxes, gt_labels, positive_keys, matching_keys, local_gt_ids):
    """Native positive links only; highest assigned-GT IoU, earliest-GT tie."""
    assert boxes.shape == (len(positive_keys), 4)
    assert torch.equal(positive_keys, positive_keys.unique(sorted=True))
    assert matching_keys.shape == local_gt_ids.shape
    target = boxes.new_zeros((len(boxes), 6), dtype=torch.float32)
    winner = torch.full((len(boxes),), -1, dtype=torch.long, device=boxes.device)
    quality = boxes.new_full((len(boxes),), -1., dtype=torch.float32)
    if not len(boxes):
        assert not len(matching_keys)
        return target, winner
    assert len(gt_boxes) and gt_boxes.shape == (len(gt_labels), 4)
    assert bool(((local_gt_ids >= 0) & (local_gt_ids < len(gt_boxes))).all())
    overlap = box_iou(boxes.float(), gt_boxes.float())
    assert torch.isfinite(overlap).all()
    for gt in range(len(gt_boxes)):
        linked = matching_keys[local_gt_ids == gt].unique(sorted=True)
        positions = torch.searchsorted(positive_keys, linked)
        assert bool((positions < len(positive_keys)).all())
        assert torch.equal(positive_keys[positions], linked)
        value = overlap[positions, gt]
        better = value > quality[positions]
        quality[positions[better]] = value[better]
        winner[positions[better]] = gt
    assert bool((winner >= 0).all()) and bool((quality >= 0).all())
    labels = gt_labels[winner].long()
    target[torch.arange(len(boxes), device=boxes.device), labels] = quality
    return target, winner


@torch.no_grad()
def same_predicted_gt_iou(boxes, gt_boxes, gt_labels, predicted):
    if not len(boxes) or not len(gt_boxes):
        return boxes.new_zeros(len(boxes), dtype=torch.float32)
    overlap = box_iou(boxes.float(), gt_boxes.float())
    overlap = torch.nan_to_num(overlap, nan=0., posinf=0., neginf=0.).clamp(0, 1)
    overlap[(boxes[:, 2:] <= boxes[:, :2]).any(1)] = 0.
    return overlap.masked_fill(predicted[:, None] != gt_labels[None, :], 0.).max(1).values


def ignore_mask(positive, same_iou, predicted, threshold=.75):
    assert positive.dtype == torch.bool and positive.ndim == 1
    assert same_iou.shape == predicted.shape == positive.shape
    assert torch.isfinite(same_iou).all() and bool(((same_iou >= 0) & (same_iou <= 1)).all())
    assert bool(((predicted >= 0) & (predicted < 6)).all())
    result = torch.zeros((len(positive), 6), dtype=torch.bool, device=positive.device)
    rows = ((~positive) & (same_iou >= threshold)).nonzero().flatten()
    result[rows, predicted[rows].long()] = True
    return result


def extraction_code(ctx):
    paths = [ctx.root / 'tools' / name for name in
        ('cache_fresh.py','calibration_common.py','distribution_features.py')]
    paths.extend(ctx.runtime / 'uod' / name for name in
        ('data.py','engine.py','geometry.py','localization_loss.py',
         'localization_distribution.py','yolov5_official.py'))
    return {str(path.relative_to(ctx.root)): sha(path) for path in paths}


def cached_marker(ctx):
    marker = ctx.cache / 'cache_complete.json'
    if not marker.exists():
        return None
    value = json.loads(marker.read_text())
    ctx.check_identity(value)
    assert value['status'] == 'completed' and value['detector_and_BN_frozen']
    assert value['extraction_code_sha256'] == extraction_code(ctx), 'Cache extractor/runtime code changed'
    for row in value['shards']:
        assert sha(ctx.cache / row['name']) == row['sha256'], 'Fresh cache shard changed'
    return value


@torch.no_grad()
def extract(ctx):
    from uod.engine import move_batch
    from uod.yolov5_official import targets_to_yolov5
    from uod.localization_loss import DistributionDetectionLoss
    from uod.geometry import undo_letterbox
    from utils.general import xywh2xyxy
    previous = cached_marker(ctx)
    if previous:
        return previous
    building = ctx.cache / 'build_identity.json'
    if building.exists():
        previous_build = json.loads(building.read_text())
        ctx.check_identity(previous_build)
        assert previous_build['extraction_code_sha256'] == extraction_code(ctx)
    else:
        assert not list(ctx.cache.glob('*_batch*.pt')), 'Unidentified partial cache'
        write(building, dict(ctx.identity(), extraction_code_sha256=extraction_code(ctx)))
    model, epoch = ctx.model()
    before = tensor_digest(model.state_dict())
    loss = DistributionDetectionLoss(model, dfl_weight=ctx.cfg['loss']['localization']['dfl_weight'])
    records = []
    totals = dict(train_pairs=0, val_pairs=0, train_rows=0, train_positive_rows=0,
                  train_unassigned_rows=0, ignored_rows=0, ignored_positive_rows=0,
                  train_image_modes=0, val_raw_candidates=0)
    sampling = hashlib.sha256()
    for split in ('train', 'val'):
        seen = []
        for batch_id, raw in enumerate(ctx.loader(split)):
            batch = move_batch(raw, ctx.device)
            with torch.autocast(device_type=ctx.device.type, dtype=torch.float16,
                                enabled=ctx.device.type == 'cuda'):
                output = model.forward_pair(batch['rgb'], batch['thermal'])
            if split == 'train':
                yolo = targets_to_yolov5(batch['targets'], batch['rgb'].shape[-2:])
                image_ids, keys, gt_ids, count = assignment_keys(output['raw_rgb'], yolo, loss)
                starts = np.cumsum([0] + [len(item['boxes']) for item in raw['original_targets']])
            parts = {key: [] for key in ('features', 'objectness', 'class_probabilities', 'scores',
                'target', 'weight', 'positive', 'ignore_mask', 'target_same_pred_GT', 'predicted_class')}
            metadata, val_rows = [], []
            for image_index, meta in enumerate(raw['meta']):
                identity = str(meta['id'])
                seen.append(identity)
                target = raw['original_targets'][image_index]
                row = dict(id=identity, meta=meta, target=target)
                if split == 'train':
                    mask = image_ids == image_index
                    matching = keys[mask]
                    local_gt_ids = gt_ids[mask] - int(starts[image_index])
                    positive = matching.unique(sorted=True)
                for mode in ('rgb', 'thermal'):
                    decoded = output['decoded_' + mode][image_index].float()
                    levels = output['raw_' + mode]
                    image_raw = torch.cat([level[image_index].reshape(-1, model.ifd.no) for level in levels], 0)
                    assert len(decoded) == len(image_raw)
                    if split == 'val':
                        features = quality_features(image_raw, model.ifd.bins, model.ifd.decode_chunk_size)
                        row[mode] = dict(decoded=decoded.cpu(), features=features.half().cpu())
                        totals['val_raw_candidates'] += len(decoded)
                    else:
                        assert len(decoded) == count
                        negative, population, negative_sha = negative_keys(count, positive, identity, mode,
                            ctx.seed, int(ctx.options['negatives_per_image_per_modality']))
                        selected = torch.cat((positive, negative))
                        features = quality_features(image_raw[selected], model.ifd.bins, model.ifd.decode_chunk_size)
                        boxes = undo_letterbox(xywh2xyxy(decoded[selected, :4]), meta)
                        gt_boxes = target['boxes'].to(ctx.device).float()
                        gt_labels = target['labels'].to(ctx.device).long()
                        assigned, winner = assigned_targets(boxes[:len(positive)], gt_boxes, gt_labels,
                            positive, matching, local_gt_ids)
                        y = torch.cat((assigned, assigned.new_zeros((len(negative), 6))), 0)
                        native_positive = torch.arange(len(selected), device=ctx.device) < len(positive)
                        classes = decoded[selected, 5:].cpu().float()
                        objectness = decoded[selected, 4].cpu().float()
                        predicted = classes.argmax(1)
                        same_iou = same_predicted_gt_iou(boxes, gt_boxes, gt_labels,
                                                        predicted.to(ctx.device)).cpu()
                        ignored = ignore_mask(native_positive.cpu(), same_iou, predicted,
                                              ctx.options['ignore_iou_threshold'])
                        weights = torch.full((len(selected),), 1. / count, dtype=torch.float32)
                        if len(negative):
                            weights[len(positive):] *= population / len(negative)
                        assert abs(float(weights.double().sum()) - 1.) < 1e-6
                        assert not bool((ignored & (y.cpu() != 0)).any())
                        values = dict(features=features.half().cpu(), objectness=objectness,
                            class_probabilities=classes, scores=objectness[:, None] * classes,
                            target=y.cpu(), weight=weights, positive=native_positive.cpu(),
                            ignore_mask=ignored, target_same_pred_GT=same_iou,
                            predicted_class=predicted.to(torch.uint8))
                        for key, value in values.items():
                            parts[key].append(value)
                        metadata.append(dict(id=identity, mode=mode, candidates=count,
                            positive_keys=positive.cpu(), positives=len(positive),
                            sampled_negatives=len(negative), background_population=population,
                            negative_keys_sha256=negative_sha,
                            resolved_local_gt_indices=winner.cpu()))
                        sampling.update((identity + '|' + mode).encode())
                        sampling.update(positive.cpu().numpy().tobytes())
                        sampling.update(negative_sha.encode())
                        totals['train_rows'] += len(selected)
                        totals['train_positive_rows'] += len(positive)
                        totals['train_unassigned_rows'] += len(negative)
                        totals['ignored_rows'] += int(ignored.sum())
                        totals['train_image_modes'] += 1
                    del image_raw, decoded, features
                if split == 'val':
                    val_rows.append(row)
            if split == 'train':
                payload = {key: torch.cat(value) for key, value in parts.items()}
                payload.update(images=metadata, image_count=len(metadata))
            else:
                payload = dict(images=val_rows)
            path = ctx.cache / ('%s_batch%04d.pt' % (split, batch_id))
            atomic_torch(path, payload)
            records.append(dict(name=path.name, sha256=sha(path), pairs=len(raw['meta'])))
            totals[split + '_pairs'] += len(raw['meta'])
            if batch_id % 10 == 0 or len(seen) == ctx.counts[split]:
                ctx.status(status='running', phase='cache_' + split, pairs=len(seen),
                           total_pairs=ctx.counts[split], sampled_rows=totals['train_rows'])
            del payload, output, batch, raw, parts
        assert seen == ctx.ids[split], 'Cache candidate identities differ from new split'
    ctx.frozen(model, before)
    result = dict(ctx.identity(), status='completed', completed_at_kst=now(), source_epoch=epoch,
        extraction_code_sha256=extraction_code(ctx),
        source_weights='EMA', detector_state_digest_before=before, detector_state_digest_after=before,
        detector_and_BN_frozen=True, base_detector_updates=0, old_confidence_prefilter_used=False,
        targets='Native YOLO-positive best assigned-GT ordinary IoU at winner class; others zero',
        ignore_policy='Unassigned and same-original-argmax-class GT IoU>=threshold; one class slot',
        fixed_loss_denominator=6, sampling_identity_sha256=sampling.hexdigest(), totals=totals, shards=records)
    assert totals['train_pairs'] == 3360 and totals['val_pairs'] == 420
    assert totals['train_image_modes'] == 6720
    write(ctx.cache / 'cache_complete.json', result)
    write(ctx.reports / 'cache.json', result)
    del model, loss
    if ctx.device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def load_training(ctx):
    marker = cached_marker(ctx)
    assert marker is not None
    fields = ('features', 'objectness', 'scores', 'class_probabilities', 'target', 'weight',
              'positive', 'ignore_mask', 'target_same_pred_GT', 'predicted_class')
    pieces = {key: [] for key in fields}
    images, ids = 0, []
    for record in marker['shards']:
        if not record['name'].startswith('train_'):
            continue
        payload = torch.load(ctx.cache / record['name'], map_location='cpu', weights_only=True, mmap=True)
        for key in fields:
            pieces[key].append(payload[key])
        images += payload['image_count']
        ids.extend(item['id'] for item in payload['images'][::2])
        native = payload['positive']
        recomputed = ignore_mask(native, payload['target_same_pred_GT'],
                                 payload['predicted_class'], ctx.options['ignore_iou_threshold'])
        assert torch.equal(recomputed, payload['ignore_mask'])
        assert not bool(payload['ignore_mask'][native].any())
        assert not bool((payload['ignore_mask'] & (payload['target'] != 0)).any())
        assert bool((payload['target'][~native] == 0).all())
        assert bool(((payload['target'] > 0).sum(1) <= 1).all())
        assert torch.equal(payload['scores'], payload['objectness'][:, None] * payload['class_probabilities'])
    assert ids == ctx.ids['train'] and images == 6720
    result = {key: torch.cat(value) for key, value in pieces.items()}
    assert abs(float(result['weight'].double().sum()) / images - 1.) < 1e-6
    assert len(result['features']) == marker['totals']['train_rows']
    assert int(result['ignore_mask'].sum()) == marker['totals']['ignored_rows']
    return result, images, marker

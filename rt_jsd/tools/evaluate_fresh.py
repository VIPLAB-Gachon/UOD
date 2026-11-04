"""Validation cache and final streaming Test for the fresh 8:1:1 experiment."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json

import torch

from calibration_common import now, sha, tensor_digest, write
from calibration_soft_nms import linear_soft_nms
from distribution_features import quality_features

MODES = ('rgb', 'thermal', 'rgbt')


def corrected_scores(head, features, decoded, chunk=32768):
    original = decoded[:, 4:5].float() * decoded[:, 5:].float()
    classes = decoded[:, 5:].float()
    device = next(head.parameters()).device
    result = torch.empty_like(original, device='cpu', dtype=torch.float32)
    for start in range(0, len(decoded), chunk):
        stop = start + chunk
        result[start:stop] = head.probability(features[start:stop].to(device).float(),
            decoded[start:stop, 4].to(device).float(), original[start:stop].to(device),
            classes[start:stop].to(device)).cpu()
    assert torch.isfinite(result).all() and bool(((result >= 0) & (result <= 1)).all())
    return result


def process_image(image, score_variants, postprocess):
    from uod.geometry import undo_letterbox
    from utils.general import xywh2xyxy
    decoded = {mode: image[mode]['decoded'].float() for mode in ('rgb', 'thermal')}
    decoded['rgbt'] = torch.cat((decoded['rgb'], decoded['thermal']), 0)
    predictions = {key: {} for key in score_variants}
    threshold = postprocess['score_threshold']
    for variant, supplied in score_variants.items():
        values = dict(supplied)
        values['rgbt'] = torch.cat((values['rgb'], values['thermal']), 0)
        for mode in MODES:
            scores, labels = values[mode].float().max(1)
            keep = scores > threshold
            pool = torch.cat((xywh2xyxy(decoded[mode][keep, :4]), scores[keep, None],
                              labels[keep, None].float()), 1)
            if len(pool):
                pool = pool[pool[:, 4].argsort(descending=True)[:postprocess.get('max_nms', 30000)]]
            det = torch.from_numpy(linear_soft_nms(pool.numpy(), postprocess['iou_threshold'],
                threshold, postprocess['max_detections']))
            predictions[variant][mode] = dict(boxes=undo_letterbox(det[:, :4], image['meta']),
                scores=det[:, 4], labels=det[:, 5].long())
    return predictions


@torch.no_grad()
def validation(ctx, head, epoch, original=False):
    from uod.metrics import DetectionAP
    head.eval()
    names = ('original', 'vfl') if original else ('vfl',)
    metrics = {name: {mode: DetectionAP(6) for mode in MODES} for name in names}
    seen = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path in sorted(ctx.cache.glob('val_batch*.pt')):
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            jobs = []
            for image in payload['images']:
                seen.append(str(image['id']))
                values = {name: {} for name in names}
                for mode in ('rgb', 'thermal'):
                    item = image[mode]
                    decoded = item['decoded'].float()
                    source = decoded[:, 4:5] * decoded[:, 5:].float()
                    calibrated = corrected_scores(head, item['features'], decoded)
                    if epoch == 0:
                        assert torch.equal(calibrated, source), 'E0 must be exact score identity'
                    values['vfl'][mode] = calibrated
                    if original:
                        values['original'][mode] = source
                jobs.append((image, pool.submit(process_image, image, values, ctx.cfg['postprocess'])))
            for image, future in jobs:
                predictions = future.result()
                for name in names:
                    for mode in MODES:
                        metrics[name][mode].update([predictions[name][mode]], [image['target']])
            ctx.status(status='running', phase='calibration_validation', epoch=epoch,
                       epochs=ctx.options['epochs'], pairs=len(seen), total_pairs=ctx.counts['val'])
            del jobs, payload
    assert seen == ctx.ids['val']
    result = {name: {mode: value.compute() for mode, value in by_mode.items()}
              for name, by_mode in metrics.items()}
    if epoch == 0 and original:
        assert result['vfl'] == result['original'], 'E0 AP must be exact original AP'
    assert all(result[name][mode]['images'] == 420 for name in names for mode in MODES)
    return result


def performance_rows(variants, classes):
    rows = []
    for variant, by_mode in variants.items():
        for mode in MODES:
            for cls, name in [(None, 'all')] + list(enumerate(classes)):
                state = by_mode[mode] if cls is None else by_mode[mode]['per_class'][str(cls)]
                prefix = 'map' if cls is None else 'ap'
                row = dict(variant=variant, mode=mode, class_name=name)
                for metric in ('50', '75', '50_95'):
                    row['AP' + metric + '_percent'] = 100 * state[prefix + metric]
                rows.append(row)
        for label, selected in (('rgb_thermal_mean', ('rgb','thermal')), ('three_mode_mean', MODES)):
            row = dict(variant=variant, mode=label, class_name='all')
            for metric in ('50', '75', '50_95'):
                row['AP' + metric + '_percent'] = 100 * sum(by_mode[mode]['map' + metric]
                    for mode in selected) / len(selected)
            rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.no_grad()
def test(ctx):
    from uod.engine import move_batch
    from uod.metrics import DetectionAP
    from score_residual import FinalClassScoreMLP
    summary_path = ctx.reports / 'test_summary.json'
    completion = json.loads((ctx.out / 'completed.json').read_text())
    ctx.check_identity(completion)
    assert completion['status'] == 'completed' and completion['epochs'] == ctx.options['epochs']
    head_path = ctx.out / 'best.pt'
    head_sha = sha(head_path)
    assert head_sha == completion['best_checkpoint_sha256']
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        ctx.check_identity(previous)
        assert previous['calibrator_checkpoint_sha256'] == head_sha and previous['status'] == 'completed'
        return previous
    state = torch.load(head_path, map_location='cpu', weights_only=True)
    ctx.check_identity(state)
    assert state['code_sha256'] == completion['code_sha256']
    for name, digest in state['code_sha256'].items():
        assert sha(ctx.root / 'tools' / name) == digest, 'Calibrator code changed after selection'
    assert state['selection_metric'] == ctx.options['best_metric']
    assert state['kind'] == 'fresh_score_calibrator'
    assert state['training_objective'] == completion['training_objective'] == ctx.options['loss']
    assert 0 <= state['epoch'] <= ctx.options['epochs']
    head = FinalClassScoreMLP().to(ctx.device)
    head.load_state_dict(state['residual_state_dict'], strict=True)
    head.requires_grad_(False).eval()
    head_before = tensor_digest(head.state_dict())
    model, source_epoch = ctx.model()
    before = tensor_digest(model.state_dict())
    variants = ('original', 'vfl')
    metrics = {name: {mode: DetectionAP(6) for mode in MODES} for name in variants}
    seen, forward_calls = [], 0
    decoded_digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for raw in ctx.loader('test'):
            batch = move_batch(raw, ctx.device)
            with torch.autocast(device_type=ctx.device.type, dtype=torch.float16,
                                enabled=ctx.device.type == 'cuda'):
                output = model.forward_pair(batch['rgb'], batch['thermal'])
            forward_calls += 1
            jobs = []
            for index, meta in enumerate(raw['meta']):
                identity = str(meta['id'])
                seen.append(identity)
                image = dict(id=identity, meta=meta, target=raw['original_targets'][index])
                scores = {name: {} for name in variants}
                for mode in ('rgb', 'thermal'):
                    decoded = output['decoded_' + mode][index].float().cpu()
                    raw_image = torch.cat([level[index].reshape(-1, model.ifd.no)
                        for level in output['raw_' + mode]], 0)
                    features = quality_features(raw_image, model.ifd.bins, model.ifd.decode_chunk_size).half()
                    scores['original'][mode] = decoded[:, 4:5] * decoded[:, 5:].float()
                    scores['vfl'][mode] = corrected_scores(head, features, decoded)
                    if state['epoch'] == 0:
                        assert torch.equal(scores['vfl'][mode], scores['original'][mode])
                    image[mode] = dict(decoded=decoded)
                    decoded_digest.update((identity + '|' + mode).encode())
                    decoded_digest.update(decoded.contiguous().numpy().tobytes())
                    del raw_image, features, decoded
                jobs.append((image, pool.submit(process_image, image, scores, ctx.cfg['postprocess'])))
            for image, future in jobs:
                predictions = future.result()
                for name in variants:
                    for mode in MODES:
                        metrics[name][mode].update([predictions[name][mode]], [image['target']])
            ctx.status(status='running', phase='final_test', pairs=len(seen), total_pairs=420,
                       selected_detector_epoch=source_epoch, selected_calibrator_epoch=state['epoch'])
            del output, batch, jobs, raw
    assert seen == ctx.ids['test'] and len(set(seen)) == 420
    values = {name: {mode: metric.compute() for mode, metric in by_mode.items()}
              for name, by_mode in metrics.items()}
    assert all(values[name][mode]['images'] == 420 for name in variants for mode in MODES)
    ctx.frozen(model, before)
    assert tensor_digest(head.state_dict()) == head_before and sha(head_path) == head_sha
    assert sha(ctx.out / 'best.pt') == completion['best_checkpoint_sha256']
    rows = performance_rows(values, ctx.classes)
    write_csv(ctx.reports / 'test_performance.csv', rows)
    comparisons = []
    by_key = {(row['variant'], row['mode'], row['class_name']): row for row in rows}
    for row in rows:
        if row['variant'] != 'vfl':
            continue
        old = by_key[('original', row['mode'], row['class_name'])]
        change = dict(mode=row['mode'], class_name=row['class_name'])
        for key in ('AP50_percent', 'AP75_percent', 'AP50_95_percent'):
            change['original_' + key] = old[key]
            change['vfl_' + key] = row[key]
            change[key.replace('_percent', '_delta_pp')] = row[key] - old[key]
        comparisons.append(change)
    write_csv(ctx.reports / 'test_comparison.csv', comparisons)
    result = dict(ctx.identity(), status='completed', completed_at_kst=now(),
        detector_epoch=source_epoch, calibrator_epoch=state['epoch'],
        calibrator_checkpoint=str(head_path), calibrator_checkpoint_sha256=head_sha,
        selection_metric=ctx.options['best_metric'], selection_tie_policy='earliest', epoch0_eligible=True,
        selected_validation=state['validation'], test_pairs=420, detector_forward_calls=forward_calls,
        detector_and_BN_frozen=True, base_detector_updates=0, test_used_for_selection=False,
        inference_has_GT_inputs=False, postprocess=ctx.cfg['postprocess'],
        shared_decoded_output_sha256=decoded_digest.hexdigest(), variants=values, calibrated_variant_key='vfl',
        calibrated_variant_training_objective=ctx.options['loss'],
        interpretation='Fresh 8:1:1 seed2026 split, not an exact published-split reproduction')
    write(summary_path, result)
    ctx.status(status='completed', phase='final_test_completed', pairs=420, total_pairs=420)
    return result

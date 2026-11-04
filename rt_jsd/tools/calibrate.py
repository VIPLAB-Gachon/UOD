"""After detector training: fresh candidates, controlled BCE/VFL MLP10E, fixed Test.

Usage: calibrate.py --config /data/.../config.json --device cuda:0 --stage all
The base detector is always frozen; only the 2182-parameter MLP receives updates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import traceback

import torch

from calibration_common import Context, atomic_torch, now, sha, tensor_digest, write
from score_loss import varifocal_elements
from score_residual import FinalClassScoreMLP


def objective(logits, target, weight, records, images, ignored, alpha=.75, gamma=2., loss_name="varifocal"):
    assert logits.shape == target.shape == ignored.shape and logits.shape[1] == 6
    assert ignored.dtype == torch.bool and weight.shape == (len(logits),)
    assert not bool((ignored & (target != 0)).any())
    if loss_name == "varifocal":
        elements = varifocal_elements(logits, target, negative_alpha=alpha, gamma=gamma)
    elif loss_name == "bce":
        elements = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    else:
        raise ValueError("Unsupported score calibrator objective: " + str(loss_name))
    masked = elements.masked_fill(ignored, 0.)
    # The fixed six-class mean and original dense HT sampling estimate are preserved.
    loss = (weight * masked.mean(1)).mean() * (records / images)
    return loss, elements


def selection_value(results, metric):
    prefix, mode, measure = metric.split('/')
    assert prefix == 'val' and measure in {'map50', 'map75', 'map50_95'}
    if mode == 'unimodal_mean':
        return .5 * (float(results['rgb'][measure]) + float(results['thermal'][measure]))
    assert mode in {'rgb', 'thermal', 'rgbt'}
    return float(results[mode][measure])


def primary(results, metric='val/unimodal_mean/map50_95'):
    return selection_value(results, metric)


def secondary(results, metric='val/rgbt/map50_95'):
    return selection_value(results, metric)


def live(ctx, best, latest=None):
    lines = ['M3FD 8:1:1 SC ' + ctx.options['loss'].upper() + ' calibration | ' + now(),
        f'Validation AP50 / AP75 / AP50-95 (%); Best = {ctx.options["best_metric"]}; earliest tie; E0 eligible.',
        f'Frozen new {ctx.cfg["training"]["epochs"]}-epoch EMA detector; MLP-only Adam0.001, 10 epochs.', '']
    for name, record in (('Original frozen detector', best['original']), ('SC ' + ctx.options['loss'].upper() + ' Best', best['best'])):
        lines.append(name + (' | epoch ' + str(record['epoch']) if 'epoch' in record else ''))
        values = record['validation']
        for mode in ('rgb', 'thermal', 'rgbt'):
            text = ' / '.join('%.3f' % (100 * values[mode]['map' + key]) for key in ('50','75','50_95'))
            lines.append('  %-7s %s' % (mode, text))
        lines.append('  RGB/T mean AP50-95 %.3f' % (100 * selection_value(values, 'val/unimodal_mean/map50_95')))
    if latest:
        lines.append('Validated through epoch %s/%s' % (latest, ctx.options['epochs']))
    temporary = ctx.root / 'calibration_live.log.writing'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(ctx.root / 'calibration_live.log')


def code_identity():
    directory = Path(__file__).resolve().parent
    names = ('calibrate.py', 'calibration_common.py', 'cache_fresh.py', 'evaluate_fresh.py',
             'score_residual.py', 'score_loss.py', 'distribution_features.py', 'calibration_soft_nms.py')
    return {name: sha(directory / name) for name in names}


def train(ctx):
    from cache_fresh import load_training
    from evaluate_fresh import validation
    completed = ctx.out / 'completed.json'
    if completed.exists():
        result = json.loads(completed.read_text())
        ctx.check_identity(result)
        assert result['status'] == 'completed' and result['code_sha256'] == code_identity()
        assert sha(ctx.out / 'best.pt') == result['best_checkpoint_sha256']
        return result
    data, images, cache = load_training(ctx)
    assert cache['detector_and_BN_frozen']
    torch.manual_seed(ctx.seed)
    head = FinalClassScoreMLP().to(ctx.device)
    initial = tensor_digest(head.state_dict())
    assert sum(parameter.numel() for parameter in head.parameters()) == 2182
    assert isinstance(head.net[1], torch.nn.ReLU)
    optimizer = torch.optim.Adam(head.parameters(), lr=ctx.options['lr'],
                                 weight_decay=ctx.options['weight_decay'])
    history_path = ctx.out / 'history.jsonl'
    last_path = ctx.out / 'last.pt'
    best_path = ctx.out / 'best.pt'
    secondary_path = ctx.out / 'best_rgbt_map50_95.pt'
    best_report = ctx.reports / 'best.json'
    marker_sha = sha(ctx.cache / 'cache_complete.json')
    code_shas = code_identity()
    if last_path.exists():
        resume = torch.load(last_path, map_location='cpu', weights_only=True)
        ctx.check_identity(resume)
        assert resume['initial_state_sha256'] == initial and resume['code_sha256'] == code_shas
        assert resume['cache_complete_sha256'] == marker_sha
        head.load_state_dict(resume['residual_state_dict'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        start = resume['epoch'] + 1
        original = resume['original_validation']
        best = resume['best_record']
        second = resume['secondary_record']
        histories = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()] if history_path.exists() else []
        # last.pt is the committed epoch; discard a partial extra journal append after a crash.
        histories = [row for row in histories if row['epoch'] < start]
        assert [row['epoch'] for row in histories] == list(range(1, start))
        history_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in histories))
    else:
        assert not history_path.exists() and not best_path.exists(), 'Uncommitted calibration files require inspection'
        controls = validation(ctx, head, 0, original=True)
        assert controls['original'] == controls['vfl']
        original = controls['original']
        best = dict(epoch=0, score=primary(original, ctx.options['best_metric']), validation=original)
        second = dict(epoch=0, score=secondary(original, ctx.options['secondary_best_metric']), validation=original)
        start = 1
        histories = []
        write(ctx.reports / 'identity.json', dict(ctx.identity(), status='passed',
            all_six_final_class_scores_bitwise_identity=True, Validation_AP_exact_identity=True,
            initial_state_sha256=initial, parameters=2182, detector_updates=0,
            cache_complete_sha256=marker_sha, code_sha256=code_shas))
        write(ctx.reports / 'validation_original.json', original)

    # The new detector and calibrator both score/filter/topK on CPU before
    # identical Soft-NMS. Replaying its selected EMA must reproduce every
    # overall and per-class Validation AP, not merely E0 vs corrected identity.
    detector_validation = ctx.root / 'training' / ('val_epoch_%03d.json' % cache['source_epoch'])
    selected_detector_result = json.loads(detector_validation.read_text())
    assert original == selected_detector_result, 'Frozen EMA original Validation differs from selected detector Validation'
    write(ctx.reports / 'detector_validation_replay.json', dict(ctx.identity(), status='passed',
        selected_detector_epoch=cache['source_epoch'], source_report=str(detector_validation),
        source_report_sha256=sha(detector_validation),
        original_Validation_AP_exact_detector_Best=True,
        overall_and_classwise_AP50_AP75_AP50_95_exact=True))

    def checkpoint(epoch, results, metric=None):
        metric = metric or ctx.options['best_metric']
        return dict(ctx.identity(), kind='fresh_score_calibrator', training_objective=ctx.options['loss'], epoch=epoch,
            residual_state_dict={key: value.detach().cpu() for key, value in head.state_dict().items()},
            optimizer=optimizer.state_dict(), validation=results, selection_metric=metric,
            best_record=best, secondary_record=second, original_validation=original,
            initial_state_sha256=initial, architecture=[27,64,6], activation='ReLU',
            calibration_config=ctx.options, detector_updates=0,
            cache_complete_sha256=marker_sha, code_sha256=code_shas)

    if start == 1:
        atomic_torch(best_path, checkpoint(0, original))
        atomic_torch(secondary_path, checkpoint(0, original, ctx.options['secondary_best_metric']))
        atomic_torch(last_path, checkpoint(0, original))
        history_path.touch()
    batch_size = int(ctx.options['batch_size'])
    rows = len(data['features'])
    assert rows == cache['totals']['train_rows'] and images == 6720
    assert data['target'].shape == data['ignore_mask'].shape == (rows, 6)
    assert int(data['ignore_mask'].sum()) == cache['totals']['ignored_rows']
    for epoch in range(start, int(ctx.options['epochs']) + 1):
        head.train()
        order = torch.randperm(rows, generator=torch.Generator().manual_seed(ctx.seed + epoch))
        weighted_loss, max_norm, steps, clipped = 0., 0., 0, 0
        for step, offset in enumerate(range(0, rows, batch_size), 1):
            ids = order[offset:offset + batch_size]
            values = {key: data[key][ids].to(ctx.device, non_blocking=True) for key in
                ('features','objectness','scores','class_probabilities','target','weight','ignore_mask')}
            optimizer.zero_grad(set_to_none=True)
            logits = head(values['features'].float(), values['objectness'], values['scores'],
                          values['class_probabilities'])
            loss, elements = objective(logits, values['target'], values['weight'], rows, images,
                values['ignore_mask'], ctx.options['vfl_alpha'], ctx.options['vfl_gamma'], ctx.options['loss'])
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10.)
            assert torch.isfinite(norm)
            max_norm = max(max_norm, float(norm))
            clipped += int(float(norm) > 10.)
            optimizer.step()
            weighted_loss += float((elements.detach().masked_fill(values['ignore_mask'], 0.).double().mean(1)
                                    * values['weight'].double()).sum())
            steps += 1
            if step == 1 or step % 25 == 0:
                ctx.status(status='running', phase='calibration_train', epoch=epoch,
                    epochs=ctx.options['epochs'], step=step, total_steps=(rows + batch_size - 1) // batch_size)
        head.eval()
        results = validation(ctx, head, epoch)['vfl']
        score, second_score = primary(results, ctx.options['best_metric']), secondary(results, ctx.options['secondary_best_metric'])
        improved, second_improved = score > best['score'], second_score > second['score']
        if improved:
            best = dict(epoch=epoch, score=score, validation=results)
        if second_improved:
            second = dict(epoch=epoch, score=second_score, validation=results)
        if improved:
            atomic_torch(best_path, checkpoint(epoch, results))
        if second_improved:
            atomic_torch(secondary_path, checkpoint(epoch, results, ctx.options['secondary_best_metric']))
        row = dict(epoch=epoch, validation=results, primary_metric=ctx.options['best_metric'], primary_score=score,
            secondary_metric=ctx.options['secondary_best_metric'], secondary_score=second_score,
            rgbt_AP50=selection_value(results, 'val/rgbt/map50'),
            mean_unimodal_AP50_95=selection_value(results, 'val/unimodal_mean/map50_95'), training_objective=ctx.options['loss'], online_masked_objective=weighted_loss / images,
            optimizer_steps=steps, preclip_norm_max=max_norm, clipped_steps=clipped)
        with history_path.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        histories.append(row)
        atomic_torch(last_path, checkpoint(epoch, results))
        report = dict(ctx.identity(), status='running', completed_epochs=epoch,
            original=dict(epoch=0, validation=original), best=best, secondary_best=second,
            source_detector_epoch=cache['source_epoch'], detector_updates=0)
        write(best_report, report)
        live(ctx, report, epoch)
        ctx.status(status='running', phase='calibration_epoch_complete', epoch=epoch,
                   epochs=ctx.options['epochs'], best_epoch=best['epoch'], best_metric=ctx.options['best_metric'], best_score=best['score'])
    assert len(histories) == ctx.options['epochs']
    ctx.check_files()
    assert sha(ctx.cache / 'cache_complete.json') == marker_sha
    completion = dict(ctx.identity(), status='completed', training_objective=ctx.options['loss'], completed_at_kst=now(), epochs=ctx.options['epochs'],
        best_epoch=best['epoch'], best_validation=best['validation'], original_validation=original,
        best_checkpoint=str(best_path), best_checkpoint_sha256=sha(best_path),
        secondary_best_epoch=second['epoch'], secondary_best_validation=second['validation'],
        selection_metric=ctx.options['best_metric'],
        secondary_selection_metric=ctx.options['secondary_best_metric'],
        secondary_checkpoint=str(secondary_path), secondary_checkpoint_sha256=sha(secondary_path), detector_updates=0, detector_and_BN_frozen=True,
        test_used_for_selection=False, cache_complete_sha256=marker_sha, code_sha256=code_shas)
    write(completed, completion)
    report = dict(ctx.identity(), status='completed', completed_epochs=ctx.options['epochs'],
        original=dict(epoch=0, validation=original), best=best, secondary_best=second,
        source_detector_epoch=cache['source_epoch'], detector_updates=0)
    write(best_report, report)
    live(ctx, report, ctx.options['epochs'])
    ctx.status(status='completed', phase='calibration_training_completed', epoch=ctx.options['epochs'],
               best_epoch=best['epoch'], best_metric=ctx.options['best_metric'], best_score=best['score'])
    return completion


def self_test():
    from cache_fresh import assigned_targets, same_predicted_gt_iou, ignore_mask, negative_keys
    from distribution_features import quality_features
    torch.manual_seed(2026)
    head = FinalClassScoreMLP()
    digest = tensor_digest(head.state_dict())
    assert digest == 'b744e9b9a9a0aa5adda493d6485efb346a9c1955665adc1394620358f625ba52'
    features = torch.rand(5, 20, requires_grad=True)
    obj = torch.tensor([0., .1, .3, .8, 1.], requires_grad=True)
    cls = torch.rand(5, 6, requires_grad=True)
    source = obj.detach()[:, None] * cls.detach()
    source[0, 0], source[-1, -1] = 0., 1.
    assert torch.equal(head.probability(features, obj, source, cls), source)
    gt_boxes = torch.tensor([[0.,0.,10.,10.], [0.,0.,10.,10.], [20.,20.,30.,30.]])
    gt_labels = torch.tensor([1,2,5])
    boxes = torch.tensor([[0.,0.,10.,10.], [20.,20.,30.,30.]])
    target, winner = assigned_targets(boxes, gt_boxes, gt_labels, torch.tensor([3,7]),
        torch.tensor([3,3,3,7]), torch.tensor([1,0,0,2]))
    assert winner.tolist() == [0,2] and target[0,1] == 1 and target[1,5] == 1
    assert int(target.count_nonzero()) == 2
    same = same_predicted_gt_iou(boxes, gt_boxes, gt_labels, torch.tensor([1,0]))
    assert same.tolist() == [1.,0.]
    positive = torch.tensor([True,False,False,False])
    mask = ignore_mask(positive, torch.tensor([1.,.75,.74999,.9]), torch.tensor([1,2,3,5]))
    assert mask.nonzero().tolist() == [[1,2],[3,5]]
    assert not bool(mask[positive].any()) and int(mask.sum()) == 2
    neg, population, negative_sha = negative_keys(1000, torch.tensor([3,7]), 'fixture', 'rgb', 2026, 128)
    other, population2, sha2 = negative_keys(1000, torch.tensor([3,7]), 'fixture', 'rgb', 2026, 128)
    assert torch.equal(neg,other) and population == population2 == 998 and negative_sha == sha2
    assert not bool(torch.isin(neg,torch.tensor([3,7])).any()) and len(neg.unique()) == 128
    weights = torch.cat((torch.full((2,), .001), torch.full((128,), 998 / (128 * 1000))))
    assert abs(float(weights.double().sum()) - 1.) < 1e-6
    logits = torch.tensor([[.2,-.3,.1,.8,-1.,1.], [.1,.7,-.1,.3,.4,.6]], requires_grad=True)
    y = torch.zeros_like(logits)
    y[0,1] = .8
    ignored = torch.zeros_like(logits, dtype=torch.bool)
    ignored[1,4] = True
    loss, elements = objective(logits,y,torch.ones(2),2,2,ignored)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
    expected = bce * torch.where(y>0,y,.75*logits.sigmoid().pow(2))
    assert torch.equal(elements,expected)
    torch.testing.assert_close(loss,expected.masked_fill(ignored,0.).mean(),rtol=0,atol=0)
    loss.backward()
    assert logits.grad[1,4] == 0 and logits.grad[1,3] != 0
    ref = logits.detach().clone().requires_grad_(True)
    (torch.nn.functional.binary_cross_entropy_with_logits(ref,y,reduction='none') *
        torch.where(y>0,y,.75*ref.sigmoid().pow(2))).masked_fill(ignored,0.).mean().backward()
    torch.testing.assert_close(logits.grad,ref.grad,rtol=0,atol=0)
    bce_logits = logits.detach().clone().requires_grad_(True)
    bce_weights = torch.tensor([2., .5])
    bce_loss, bce_elements = objective(bce_logits, y, bce_weights, 6, 3, ignored,
        loss_name='bce')
    bce_expected = torch.nn.functional.binary_cross_entropy_with_logits(bce_logits, y, reduction='none')
    assert torch.equal(bce_elements, bce_expected)
    expected_weighted_bce = (bce_weights * bce_expected.masked_fill(ignored, 0.).mean(1)).mean() * 2
    torch.testing.assert_close(bce_loss, expected_weighted_bce, rtol=0, atol=0)
    bce_loss.backward()
    assert bce_logits.grad[1, 4] == 0 and bce_logits.grad[1, 3] != 0
    bce_ref = logits.detach().clone().requires_grad_(True)
    ref_elements = torch.nn.functional.binary_cross_entropy_with_logits(bce_ref, y, reduction='none')
    (bce_weights * ref_elements.masked_fill(ignored, 0.).mean(1)).mean().mul(2).backward()
    torch.testing.assert_close(bce_logits.grad, bce_ref.grad, rtol=0, atol=0)
    target2 = torch.zeros(5,6)
    target2[2,1] = .8
    head(features,obj,source,cls).square().mean().backward()
    assert features.grad is None and obj.grad is None and cls.grad is None
    assert head.net[-1].weight.grad is not None and torch.isfinite(head.net[-1].weight.grad).all()
    raw = torch.randn(7,107,requires_grad=True)
    stats = quality_features(raw)
    top = raw.detach()[:,:100].reshape(-1,4,25).softmax(-1).sort(descending=True).values[...,:4]
    assert torch.equal(stats,torch.cat((top,top.mean(-1,keepdim=True)),-1).flatten(1))
    assert not stats.requires_grad
    # Strict improvement preserves epoch0/earliest winner on an exact tie.
    winner_epoch, best_score = 0, .8
    for epoch, score in [(1,.8),(2,.81),(3,.81)]:
        if score > best_score:
            winner_epoch,best_score = epoch,score
    assert winner_epoch == 2
    metric_probe = {mode: {'map50': ap50, 'map50_95': ap95} for mode, ap50, ap95 in
        [('rgb', .9, .6), ('thermal', .8, .5), ('rgbt', .93, .64)]}
    assert abs(primary(metric_probe, 'val/unimodal_mean/map50_95') - .55) < 1e-9
    assert secondary(metric_probe, 'val/rgbt/map50_95') == .64
    return dict(status='passed', initial_state_sha256=digest, parameters=2182,
        tests=['fresh_initialization_exact_successful_MLP', 'zero_residual_all_score_identity',
               'assigned_GT_collision_highest_IoU_earliest_tie', 'same_predicted_class_overlap',
               'single_slot_ignore_threshold_positive_protection', 'deterministic_disjoint_sampling',
               'HT_weights_sum_one', 'VFL_values_and_focal_weight_gradient',
               'BCE_same_targets_ignore_HT_normalization_and_gradient',
               'ignored_gradient_zero_fixed_six_column_mean', 'all_detector_inputs_detached',
               'distribution_top4_and_mean_features', 'epoch0_eligible_earliest_tie',
               'primary_unimodal_AP95_and_fusion_AP95_selection'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=('all','cache','train','test','self-test'), default='all')
    args = parser.parse_args()
    root = Path(args.config).resolve().parent
    if args.stage == 'self-test':
        result = self_test()
        write(root / 'reports/calibration/self_tests.json', result)
        print(json.dumps(result, ensure_ascii=False))
        return
    ctx = Context(args.config, args.device)
    try:
        from uod.engine import seed_all
        from uod.reproducibility import configure_determinism
        configure_determinism(bool(ctx.cfg.get('deterministic', True)))
        seed_all(ctx.seed)
        torch.set_num_threads(2)
        assert ctx.device.type == 'cuda' and torch.cuda.is_available()
        ctx.status(status='running', phase='starting', stage=args.stage)
        if args.stage in ('all','cache'):
            from cache_fresh import extract
            extract(ctx)
        if args.stage in ('all','train'):
            train(ctx)
        if args.stage in ('all','test'):
            from evaluate_fresh import test
            test(ctx)
    except BaseException as error:
        ctx.status(status='failed', phase='failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

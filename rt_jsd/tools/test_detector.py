"""Evaluate the validation-selected EMA without a score calibrator.

This is the completion path for a configured GT-only/JSD detector-only run.
It uses the same paired loader, numerical settings and Soft-NMS as training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from train_ddp import (build_dataset, make_loader, move_batch, read, seed_all,
    soft_postprocess, stamp, torch, write)
from calibration_common import Context, tensor_digest
from uod.metrics import DetectionAP
from uod.reproducibility import configure_determinism, file_sha256


@torch.no_grad()
def run(config, device):
    cfg = read(config)
    torch.set_num_threads(2)
    configure_determinism(bool(cfg['deterministic']))
    seed_all(int(cfg['seed']))
    ctx = Context(config, device)
    model, selected_epoch = ctx.model()
    before = tensor_digest(model.state_dict())
    dataset = build_dataset(cfg, 'test', False)
    assert len(dataset) == cfg['experiment']['test_pairs'] == 420
    loader = make_loader(dataset, int(cfg['training']['val_batch_size']), 0,
        False, ctx.device, torch.Generator().manual_seed(int(cfg['seed']) + 1))
    metrics = {mode: DetectionAP(model.nc) for mode in ('rgb', 'thermal', 'rgbt')}
    seen = []
    for raw in loader:
        batch = move_batch(raw, ctx.device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=bool(cfg['training']['amp'])):
            output = model.forward_pair(batch['rgb'], batch['thermal'])
        rgb, thermal = output['decoded_rgb'].float(), output['decoded_thermal'].float()
        for mode, decoded in dict(rgb=rgb, thermal=thermal, rgbt=torch.cat((rgb, thermal), 1)).items():
            formatted = [soft_postprocess(image, meta, cfg['postprocess'])
                for image, meta in zip(decoded, raw['meta'], strict=True)]
            metrics[mode].update(formatted, raw['original_targets'])
        seen.extend(str(meta['id']) for meta in raw['meta'])
        print(f'Test {len(seen)}/{len(dataset)} pairs | selected EMA epoch {selected_epoch}', flush=True)
        del batch, output, rgb, thermal
    assert seen == ctx.ids['test']
    values = {mode: metric.compute() for mode, metric in metrics.items()}
    ctx.frozen(model, before)
    report = dict(status='completed', completed_at_kst=stamp(), config=str(Path(config).resolve()),
        config_sha256=file_sha256(config), source_checkpoint=str(ctx.checkpoint),
        source_checkpoint_sha256=ctx.detector_sha, detector_epoch=selected_epoch,
        detector_selection_metric=cfg['training']['best_metric'], calibrator_enabled=False,
        test_used_for_selection=False, test_pairs=len(dataset), input_size=cfg['data']['image_size'],
        postprocess=cfg['postprocess'], test=values,
        means={metric: dict(rgb_thermal=.5 * (values['rgb'][metric] + values['thermal'][metric]),
            three_modes=sum(values[mode][metric] for mode in metrics) / 3)
            for metric in ('map50', 'map75', 'map50_95')})
    write(ctx.root / 'reports/detector_test_summary.json', report)
    rows = [dict(mode=mode, **{metric: 100 * result[metric] for metric in ('map50', 'map75', 'map50_95')})
        for mode, result in values.items()]
    for name, key in (('rgb_thermal_mean', 'rgb_thermal'), ('three_mode_mean', 'three_modes')):
        rows.append(dict(mode=name, **{metric: 100 * report['means'][metric][key]
            for metric in ('map50', 'map75', 'map50_95')}))
    with (ctx.root / 'reports/detector_test_performance.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['mode', 'map50', 'map75', 'map50_95'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run(args.config, args.device)

"""Fresh 8:1:1 calibration context; no historical checkpoint/cache imports."""
from __future__ import annotations
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import torch


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    torch.save(value, temporary)
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Context:
    def __init__(self, config, device='cuda:0'):
        self.path = Path(config).resolve()
        self.root = self.path.parent
        self.cfg = json.loads(self.path.read_text())
        self.options = self.cfg['calibration']
        self.device = torch.device(device)
        self.runtime = self.root / 'runtime'
        sys.path.insert(0, str(self.runtime))
        self.seed = int(self.options.get('seed', self.cfg['seed']))
        self.base_batch = int(self.options.get('forward_batch_size', 8))
        self.workers = int(self.options.get('workers', 0))
        self.checkpoint = Path(self.options['detector_checkpoint']).resolve()
        assert self.checkpoint == (self.root / 'training/best.pt').resolve(), 'New detector Best required'
        assert self.options['frozen_detector'] and self.options['use_original_class_probabilities']
        assert self.options['architecture'] == '27-64-ReLU-6'
        assert self.options['best_metric'] in {'val/unimodal_mean/map50_95', 'val/rgbt/map50_95'}
        assert self.options['secondary_best_metric'] == 'val/rgbt/map50_95'
        assert self.options['include_epoch_zero'] and self.options['tie_policy'] == 'earliest'
        assert self.options['optimizer'] == 'Adam'
        assert self.options['loss'] in {'bce', 'varifocal'}
        self.classes = self.cfg['data']['class_names']
        assert self.classes == ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
        self.counts = {key: int(self.cfg['experiment'][key + '_pairs']) for key in ('train', 'val', 'test')}
        assert self.counts == {'train': 3360, 'val': 420, 'test': 420}
        self.cache = self.root / 'cache/fresh_calibration'
        self.out = self.root / 'calibrator'
        self.reports = self.root / 'reports/calibration'
        for path in (self.cache, self.out, self.reports):
            path.mkdir(parents=True, exist_ok=True)
        self.config_sha = sha(self.path)
        self.ids, self.manifest_shas = {}, {}
        for split in ('train', 'val', 'test'):
            path = Path(self.cfg['data'][split + '_manifest']).resolve()
            assert path.parent.name == 'm3fd_uod_811_seed2026', 'Fresh split manifests required'
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            ids = [str(row['id']) for row in rows]
            assert len(ids) == self.counts[split] == len(set(ids)), 'Manifest count/identity mismatch'
            self.ids[split] = ids
            self.manifest_shas[split] = sha(path)
        for left, right in (('train','val'), ('train','test'), ('val','test')):
            assert not set(self.ids[left]) & set(self.ids[right]), 'Split overlap'
        self.detector_sha = sha(self.checkpoint) if self.checkpoint.exists() else None

    def identity(self):
        assert self.detector_sha is not None, 'Detector checkpoint not yet available'
        return dict(config_sha256=self.config_sha, source_checkpoint=str(self.checkpoint),
                    source_checkpoint_sha256=self.detector_sha, manifest_sha256=self.manifest_shas)

    def status(self, **values):
        info = dict(updated_at_kst=now(), **values)
        write(self.out / 'status.json', info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    def check_identity(self, value):
        for key, expected in self.identity().items():
            assert value[key] == expected, 'Fresh pipeline identity mismatch: ' + key

    def check_files(self):
        assert sha(self.path) == self.config_sha
        assert sha(self.checkpoint) == self.detector_sha
        for split in ('train', 'val', 'test'):
            assert sha(self.cfg['data'][split + '_manifest']) == self.manifest_shas[split]

    def model(self):
        from uod.yolov5_official import OfficialYOLOv5UOD
        from uod.localization_distribution import DistributionDetect
        from train_yolov5_uod import scaled_yolov5_hyp
        self.check_files()
        training_status = json.loads((self.root / 'training/status.json').read_text())
        assert training_status['status'] == 'completed'
        assert training_status['completed_epoch'] == training_status['target_epochs'] == self.cfg['training']['epochs']
        assert training_status['best_metric'] == self.cfg['training']['best_metric']
        checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        assert checkpoint['config_sha256'] == self.config_sha
        assert checkpoint['best_score'] == checkpoint['best_scores'][self.cfg['training']['best_metric']]
        cfg = copy.deepcopy(checkpoint['config'])
        assert cfg['data'] == self.cfg['data'], 'Detector was trained on different data'
        assert cfg['model'] == self.cfg['model']
        assert cfg['loss'] == self.cfg['loss']
        for key in ('epochs', 'optimizer', 'batch_size', 'lr0', 'lr_schedule_epochs'):
            assert cfg['training'][key] == self.cfg['training'][key]
        assert cfg['training']['epochs'] == self.cfg['training']['epochs'] and cfg['training']['batch_size'] == self.cfg['training']['batch_size']
        assert cfg['training']['lr0'] == .01 and cfg['training']['optimizer'] == 'SGD'
        assert 0 <= int(checkpoint['epoch']) < cfg['training']['epochs']
        model = OfficialYOLOv5UOD.from_pretrained(cfg['model']['weights'], self.classes,
            use_nfe=False, separate_ife=True, feature_objective=cfg['loss']['feature_objective'])
        model.ifd = DistributionDetect(model.ifd, **cfg['model']['localization_distribution'])
        model.stride = model.ifd.stride
        model.hyp = scaled_yolov5_hyp(cfg, model)
        model.load_state_dict(checkpoint['ema_state_dict'], strict=True)
        model.requires_grad_(False).to(self.device).eval()
        return model, int(checkpoint['epoch']) + 1

    def loader(self, split):
        from uod.engine import build_dataset, make_loader
        dataset = build_dataset(self.cfg, split, False)
        assert len(dataset) == self.counts[split]
        loader = make_loader(dataset, self.base_batch, self.workers, False, self.device,
                             torch.Generator().manual_seed(self.seed + 1))
        if self.workers:
            # Do not fork an already initialized CUDA/OpenMP parent process.
            loader.multiprocessing_context = 'spawn'
        return loader

    def frozen(self, model, before):
        assert not model.training and all(not module.training for module in model.modules())
        assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
        assert tensor_digest(model.state_dict()) == before, 'Detector or BN changed'
        self.check_files()

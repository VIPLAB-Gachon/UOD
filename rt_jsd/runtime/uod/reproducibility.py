"""Explicit RNG/backend state for matched experiment forks."""
from __future__ import annotations
import os
import random
import hashlib
import copy
import numpy as np
import torch


def configure_determinism(enabled: bool):
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in {':4096:8', ':16:8'}:
            raise ValueError('Deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = False
    return {'enabled': enabled, 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'torch': str(torch.__version__)}


def numpy_rng_state():
    name, keys, position, gaussian, cached = np.random.get_state()
    return {'name': name, 'keys': keys.tolist(), 'position': position,
            'has_gaussian': gaussian, 'cached_gaussian': cached}


def restore_rng(checkpoint, train_generator, val_generator, probe_generator=None, *, strict=False):
    if strict and any(k not in checkpoint for k in ('numpy_rng', 'probe_loader_rng', 'backend')):
        raise ValueError('A common fork requires a complete version-2 RNG/backend checkpoint')
    random.setstate(checkpoint['python_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    if 'numpy_rng' in checkpoint:
        state = checkpoint['numpy_rng']
        np.random.set_state((state['name'], np.asarray(state['keys'], dtype=np.uint32),
                            state['position'], state['has_gaussian'], state['cached_gaussian']))
    train_generator.set_state(checkpoint['loader_rng'])
    val_generator.set_state(checkpoint['val_loader_rng'])
    if probe_generator is not None and checkpoint.get('probe_loader_rng') is not None:
        probe_generator.set_state(checkpoint['probe_loader_rng'])
    if torch.cuda.is_available() and checkpoint.get('cuda_rng') is not None:
        saved = checkpoint['cuda_rng']
        if len(saved) != torch.cuda.device_count():
            if strict:
                raise ValueError('CUDA RNG device count differs from the common checkpoint')
        else:
            torch.cuda.set_rng_state_all(saved)


def validate_fork_config(source, target):
    """Validate this user-authorized C recovery: micro5->2 and val16->4."""
    def normalized(cfg):
        cfg = copy.deepcopy(cfg)
        cfg.pop('runtime', None)
        cfg.pop('device', None)
        cfg['training'].pop('out_dir', None)
        cfg['training'].pop('epochs', None)
        cfg['loss'].pop('alpha', None)
        cfg['loss'].setdefault('localization', {}).pop('edge_gate', None)
        cfg['loss'].setdefault('js_alignment', {}).pop('region', None)
        return cfg
    left, right = normalized(source), normalized(target)
    for field, permitted in {'micro_batch_size': (5, 2), 'val_batch_size': (16, 4)}.items():
        pair = (left['training'][field], right['training'][field])
        if pair[0] != pair[1]:
            if pair != permitted:
                raise ValueError(f'Unauthorized batch override for {field}: {pair}')
            right['training'][field] = left['training'][field]
    if left != right:
        raise ValueError('Fork configurations differ beyond alpha, edge_gate, alignment region, duration, device, or output')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def batch_fingerprint(batch):
    """Bounded auditing only; hashes actual augmented tensors and GT, on CPU."""
    digest = hashlib.sha256()
    tensors = [batch[k] for k in ('rgb', 'thermal', 'valid_mask')]
    for target in batch['targets']:
        tensors.extend(target[k] for k in ('boxes', 'labels'))
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return {'ids': [meta['id'] for meta in batch['meta']], 'sha256': digest.hexdigest()}


def state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(state_equal(a,b) for a,b in zip(left,right))
    return left == right


def verify_restored_fork(source_path, restored):
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    fields = ['state_dict', 'ema_state_dict', 'optimizer', 'scheduler', 'scaler',
              'epoch', 'global_update', 'ema_updates', 'best_score', 'best_scores',
              'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'loader_rng',
              'val_loader_rng', 'probe_loader_rng', 'backend']
    checks = {key: state_equal(source[key], restored[key]) for key in fields}
    if not all(checks.values()):
        raise RuntimeError('Fork state mismatch: '+', '.join(key for key,ok in checks.items() if not ok))
    return {'status': 'passed', 'state_equal': checks,
            'completed_epoch': int(source['epoch'])+1, 'global_update': source['global_update']}

"""Vectorized historical within-GT IoU-gap hinge, with class diagnostics."""
import torch


def within_gt_confidence_ranking(scores, quality, keys, gt_ids, valid,
                                 gt_batch_ids, batch_size, gt_classes, class_ids):
    n = scores.numel()
    if any(v.ndim != 1 or v.numel() != n for v in (scores, quality, keys, gt_ids, valid)):
        raise ValueError('Candidate vectors must be equally sized one-dimensional tensors')
    if any(v.dtype != torch.long for v in (keys, gt_ids, gt_batch_ids, gt_classes)):
        raise TypeError('Keys and GT/image/class IDs must be int64')
    if valid.dtype != torch.bool or gt_batch_ids.ndim != 1 or batch_size < 1:
        raise ValueError('Invalid validity mask, GT owners or batch size')
    num_gt = len(gt_batch_ids)
    if gt_classes.shape != gt_batch_ids.shape:
        raise ValueError('GT classes must align with GT owners')
    scores, quality = scores.float(), quality.detach().float()
    zero = scores.sum() * 0.
    counts = {name: zero.detach().clone() for name in ('candidates', 'ambiguous_candidates',
        'invalid_candidates', 'ordered_pairs', 'inverted_pairs', 'equal_score_pairs',
        'margin_violations', 'gt_with_pairs')}
    diag = {f'class_{c}_{key}': zero.detach().clone() for c in class_ids
            for key in ('contribution', 'gt_per_pair', 'ordered_pairs_per_pair', 'gt_with_pairs_per_pair')}
    for c in class_ids:
        diag[f'class_{c}_gt_per_pair'] = (gt_classes == c).sum().float() / batch_size
    if not n:
        return zero, dict(diag, **{k + '_per_pair': v for k, v in counts.items()})
    if not num_gt or int(gt_ids.min()) < 0 or int(gt_ids.max()) >= num_gt:
        raise ValueError('Candidate GT ID outside target rows')
    if int(gt_batch_ids.min()) < 0 or int(gt_batch_ids.max()) >= batch_size:
        raise ValueError('GT image ID outside batch')

    # Count distinct GT links per candidate BEFORE applying class/validity mask.
    order = torch.argsort(keys * (num_gt + 1) + gt_ids, stable=True)
    k, g = keys[order], gt_ids[order]
    unique = torch.ones_like(k, dtype=torch.bool)
    unique[1:] = (k[1:] != k[:-1]) | (g[1:] != g[:-1])
    order, k = order[unique], k[unique]
    _, assignments = torch.unique_consecutive(k, return_counts=True)
    counts['ambiguous_candidates'] = (assignments > 1).sum().float()
    order = order[torch.repeat_interleave(assignments, assignments, output_size=len(order)) == 1]
    counts['invalid_candidates'] = (~valid[order]).sum().float()
    order = order[valid[order]]
    counts['candidates'] = scores.new_tensor(len(order))
    if not len(order):
        return zero, dict(diag, **{k + '_per_pair': v.detach() / batch_size for k, v in counts.items()})

    order = order[torch.argsort(gt_ids[order], stable=True)]
    owners = gt_ids[order]
    sizes = torch.bincount(owners, minlength=num_gt)
    width = int(sizes.max())
    starts = sizes.cumsum(0) - sizes
    columns = torch.arange(len(order), device=scores.device) - torch.repeat_interleave(starts, sizes, output_size=len(order))
    s = scores.new_zeros((num_gt, width)).index_put((owners, columns), scores[order])
    q = quality.new_zeros((num_gt, width)).index_put((owners, columns), quality[order])
    filled = torch.arange(width, device=scores.device)[None, :] < sizes[:, None]
    dq, ds = q[:, :, None] - q[:, None, :], s[:, :, None] - s[:, None, :]
    selected = (dq > 0) & filled[:, :, None] & filled[:, None, :]
    pairs = selected.sum((1, 2))
    hinge = (dq - ds).clamp_min(0)
    groups = (hinge * selected).sum((1, 2)) / pairs.clamp_min(1)
    denominators = torch.bincount(gt_batch_ids, minlength=batch_size).clamp_min(1)
    weights = 1. / denominators[gt_batch_ids].float() / batch_size
    loss = (groups * weights).sum()
    counts.update(ordered_pairs=pairs.sum(), inverted_pairs=((ds.detach() < 0) & selected).sum(),
        equal_score_pairs=((ds.detach() == 0) & selected).sum(),
        margin_violations=((hinge.detach() > 0) & selected).sum(), gt_with_pairs=(pairs > 0).sum())
    for c in class_ids:
        mask = gt_classes == c
        diag[f'class_{c}_contribution'] = (groups.detach() * weights * mask).sum()
        diag[f'class_{c}_ordered_pairs_per_pair'] = pairs[mask].sum().float() / batch_size
        diag[f'class_{c}_gt_with_pairs_per_pair'] = (pairs[mask] > 0).sum().float() / batch_size
    return loss, dict(diag, **{k + '_per_pair': v.detach().float() / batch_size for k, v in counts.items()})

"""Deterministic class-aware linear Soft-NMS using continuous YOLO box IoU.

Algorithm reference: Bodla et al., ICCV 2017, author implementation:
https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx
Coordinates use the current detector's continuous convention, without +1.
The input is the exact same confidence-filtered/top-30000 pool as Hard-NMS.
There are no GT inputs, model parameters, or test-adapted thresholds.
"""
import numpy as np


def linear_soft_nms(detections, iou_threshold=.6, score_threshold=.001, max_det=300,
                    return_indices=False):
    d = np.asarray(detections, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != 6:
        raise ValueError('Expected Nx6 [x1,y1,x2,y2,score,class] detections')
    if not 0 <= iou_threshold <= 1 or not 0 <= score_threshold <= 1 or max_det < 1:
        raise ValueError('Invalid Soft-NMS thresholds or output cap')
    if not len(d):
        return (d.copy(), np.empty(0, dtype=np.int64)) if return_indices else d.copy()
    scores = d[:, 4].copy()
    widths = np.maximum(d[:, 2] - d[:, 0], 0)
    heights = np.maximum(d[:, 3] - d[:, 1], 0)
    areas = widths * heights
    selected = []; indices = []
    for _ in range(min(max_det, len(d))):
        index = int(scores.argmax())
        value = float(scores[index])
        if value <= score_threshold:
            break
        best = d[index].copy(); best[4] = value
        selected.append(best)
        indices.append(index)
        scores[index] = -np.inf
        others = np.flatnonzero((scores > score_threshold) & (d[:, 5] == best[5]))
        if not len(others):
            continue
        wh = np.maximum(np.minimum(d[others, 2:4], best[2:4])
                        - np.maximum(d[others, :2], best[:2]), 0)
        intersection = wh[:, 0] * wh[:, 1]
        union = areas[others] + areas[index] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        decay = np.where(iou > iou_threshold, 1. - iou, 1.).astype(np.float32)
        scores[others] *= decay
    result = np.stack(selected) if selected else np.empty((0, 6), dtype=np.float32)
    return (result, np.asarray(indices, dtype=np.int64)) if return_indices else result

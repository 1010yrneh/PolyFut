"""Interval recall/precision metrics for possession segment evaluation."""

from __future__ import annotations


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    iv = sorted(intervals, key=lambda x: x[0])
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def union_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in _merge_intervals(intervals))


def overlap_duration(
    pred: list[tuple[float, float]],
    gt: list[tuple[float, float]],
) -> float:
    """Seconds of ground-truth time covered by predicted union."""
    if not pred or not gt:
        return 0.0
    pred_m = _merge_intervals(pred)
    gt_m = _merge_intervals(gt)
    total = 0.0
    for gs, ge in gt_m:
        for ps, pe in pred_m:
            s = max(gs, ps)
            e = min(ge, pe)
            if e > s:
                total += e - s
    return total


def interval_metrics(
    pred: list[tuple[float, float]],
    gt: list[tuple[float, float]],
) -> dict:
    gt_sec = union_duration(gt)
    pred_sec = union_duration(pred)
    overlap = overlap_duration(pred, gt)
    recall = overlap / gt_sec if gt_sec > 0 else 0.0
    precision = overlap / pred_sec if pred_sec > 0 else 0.0
    return {
        "interval_recall": round(recall, 4),
        "interval_precision": round(precision, 4),
        "gt_seconds": round(gt_sec, 2),
        "pred_seconds": round(pred_sec, 2),
        "overlap_seconds": round(overlap, 2),
        "n_pred_segments": len(pred),
        "n_gt_intervals": len(gt),
        "passes_75_recall": recall >= 0.75,
    }

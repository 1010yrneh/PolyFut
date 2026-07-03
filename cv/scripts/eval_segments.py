"""Evaluate clip_segments.json against hand-labelled ground truth.

Usage:
    python cv/scripts/eval_segments.py \\
        --segments exports/run1/clip_segments.json \\
        --labels sample_data/labels.json \\
        --out exports/run1/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CV_SRC = ROOT / "cv" / "src"
if str(CV_SRC) not in sys.path:
    sys.path.insert(0, str(CV_SRC))

from polyfut_cv.eval import interval_metrics  # noqa: E402


def _load_segments(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", data if isinstance(data, list) else [])
    out: list[tuple[float, float]] = []
    for s in segs:
        out.append((float(s["start"]), float(s["end"])))
    return out


def _load_gt(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("true_possession_intervals", [])
    return [(float(a), float(b)) for a, b in raw]


def main() -> int:
    ap = argparse.ArgumentParser(description="Score pipeline output vs labels.json")
    ap.add_argument("--segments", required=True, help="clip_segments.json path")
    ap.add_argument("--labels", default=str(ROOT / "sample_data" / "labels.json"))
    ap.add_argument("--out", default=None, help="Write JSON report here")
    args = ap.parse_args()

    seg_path = Path(args.segments)
    labels_path = Path(args.labels)
    if not seg_path.exists():
        raise SystemExit(f"Segments not found: {seg_path}")
    if not labels_path.exists():
        raise SystemExit(f"Labels not found: {labels_path} (copy labels.template.json)")

    pred = _load_segments(seg_path)
    gt = _load_gt(labels_path)
    metrics = interval_metrics(pred, gt)
    report = {
        "segments_path": str(seg_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        **metrics,
    }

    print("\n=== Segment evaluation ===")
    print(f"interval_recall:    {metrics['interval_recall']:.1%}  (target >= 75%)")
    print(f"interval_precision: {metrics['interval_precision']:.1%}")
    print(f"gt_seconds:         {metrics['gt_seconds']}")
    print(f"pred_seconds:       {metrics['pred_seconds']}")
    print(f"overlap_seconds:    {metrics['overlap_seconds']}")
    print(f"segments:           {metrics['n_pred_segments']} pred / {metrics['n_gt_intervals']} gt")
    print(f"passes_75_recall:   {metrics['passes_75_recall']}")

    out_path = Path(args.out) if args.out else seg_path.parent / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport -> {out_path}")
    return 0 if metrics["passes_75_recall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

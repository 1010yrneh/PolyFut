"""polyfut-cv command line: run the pipeline from a video + colour reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyfut_cv.color_classify import ColorRef
from polyfut_cv.pipeline import PipelineConfig, run_pipeline


def _load_color_ref(args: argparse.Namespace) -> ColorRef:
    if args.color_json:
        return ColorRef.from_json(args.color_json)
    if args.hsv:
        h, s, v = (float(x) for x in args.hsv.split(","))
        return ColorRef(h=h, s=s, v=v)
    raise SystemExit("Provide --color-json or --hsv H,S,V (OpenCV ranges: H 0-179, S/V 0-255)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="polyfut-cv", description="Team-possession clip finder")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Full pipeline -> clip_segments.json")
    r.add_argument("--video", required=True)
    r.add_argument("--out", default="exports/run1")
    r.add_argument("--color-json", help="ColorRef JSON from sample_color.py / seed screen")
    r.add_argument("--hsv", help="Direct HSV 'H,S,V' (OpenCV ranges)")
    r.add_argument("--weights", default="yolov8n.pt")
    r.add_argument("--imgsz", type=int, default=1280)
    r.add_argument("--stride", type=int, default=6)
    r.add_argument("--conf", type=float, default=0.20)
    r.add_argument("--device", default="cpu")
    r.add_argument("--max-minutes", type=float, default=90.0)
    r.add_argument("--no-cut-gate", action="store_true")
    r.add_argument("--no-blur-gate", action="store_true")
    r.add_argument("--pitch-mask", action="store_true")
    r.add_argument("--gap-merge", type=float, default=7.0, help="Play-bucket merge gap (s)")

    args = p.parse_args(argv)

    if args.cmd == "run":
        color_ref = _load_color_ref(args)
        cfg = PipelineConfig(
            stride=args.stride,
            imgsz=args.imgsz,
            conf=args.conf,
            weights=args.weights,
            device=args.device,
            max_minutes=args.max_minutes,
            use_cut_gate=not args.no_cut_gate,
            use_blur_gate=not args.no_blur_gate,
            use_pitch_mask=args.pitch_mask,
        )
        cfg.export.gap_merge_sec = args.gap_merge

        def _progress(frac: float, msg: str) -> None:
            sys.stdout.write(f"\r[{frac*100:5.1f}%] {msg:<40}")
            sys.stdout.flush()

        meta = run_pipeline(args.video, color_ref, args.out, cfg, progress=_progress)
        print()
        print(f"Done: {meta['n_segments']} plays -> {meta['clip_segments_path']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

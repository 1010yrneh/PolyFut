"""One-time: export yolov8n.pt -> OpenVINO IR (FP16) for a 3-5x CPU speedup.

Then run the pipeline with --weights pointing at the exported directory:
    polyfut-cv run --video match.mp4 --color-json color_ref.json \
        --weights yolov8n_openvino_model

Requires: pip install openvino
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--int8", action="store_true", help="INT8 quantization (validate accuracy after)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics")
    try:
        import openvino  # noqa: F401
    except ImportError:
        raise SystemExit("pip install openvino")

    model = YOLO(args.weights)
    out = model.export(format="openvino", imgsz=args.imgsz, half=not args.int8, int8=args.int8)
    print(f"Exported -> {out}")
    print(f"Use: --weights {Path(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

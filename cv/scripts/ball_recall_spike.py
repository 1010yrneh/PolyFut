"""Session 0: ball-detection feasibility spike.

Run stock YOLOv8n over a sample clip and count how often the ball is detected,
optionally split by camera angle. THIS IS THE DECISION GATE: if the ball is rarely
seen, the whole possession pipeline will be empty/wrong, and you should escalate
(larger imgsz, SAHI tiles, or a soccer-ball-specific model) before building more.

Usage:
    python cv/scripts/ball_recall_spike.py --video sample_data/clip.mp4
    python cv/scripts/ball_recall_spike.py --video sample_data/clip.mp4 \
        --angles sample_data/angle_ranges.yaml --imgsz 1280 --conf 0.15

angle_ranges.yaml (optional):
    elevated: [[0, 30]]      # second ranges that are the elevated camera
    sideline: [[30, 60]]     # second ranges that are the ground/sideline camera
"""

from __future__ import annotations

import argparse
from pathlib import Path

CLASS_BALL = 32


def _load_angles(path: str | None):
    if not path:
        return {}
    import yaml  # lazy
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {k: [tuple(r) for r in v] for k, v in data.items()}


def _angle_of(t: float, angles: dict) -> str:
    for name, ranges in angles.items():
        for s, e in ranges:
            if s <= t <= e:
                return name
    return "unlabelled"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--angles", default=None)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    angles = _load_angles(args.angles)
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    stats: dict[str, dict[str, int]] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            t = idx / fps
            angle = _angle_of(t, angles)
            res = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                classes=[CLASS_BALL], verbose=False)[0]
            seen = res.boxes is not None and len(res.boxes) > 0
            d = stats.setdefault(angle, {"frames": 0, "ball": 0})
            d["frames"] += 1
            d["ball"] += 1 if seen else 0
        idx += 1
    cap.release()

    print("\n=== Ball recall spike ===")
    print(f"video={args.video}  imgsz={args.imgsz}  conf={args.conf}  stride={args.stride}")
    print(f"{'angle':<14}{'frames':>8}{'ball':>8}{'recall':>9}")
    for angle, d in stats.items():
        rec = d["ball"] / d["frames"] if d["frames"] else 0.0
        print(f"{angle:<14}{d['frames']:>8}{d['ball']:>8}{rec*100:>8.1f}%")

    sideline = stats.get("sideline")
    if sideline and sideline["frames"]:
        rec = sideline["ball"] / sideline["frames"]
        print("\nDECISION GATE (sideline):", "PASS" if rec >= 0.5 else "INVESTIGATE (try SAHI / soccer-ball weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

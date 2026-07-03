"""Click your team's jersey on the first frame -> write a ColorRef JSON.

Requires a desktop OpenCV window (pip install opencv-python). The browser seed
screen does the same thing without a terminal.

Usage:
    python cv/scripts/sample_color.py --video sample_data/clip.mp4 --out sample_data/color_ref.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyfut_cv.color_classify import sample_color_ref  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="color_ref.json")
    args = ap.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read frame")

    picked: dict = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked["xy"] = (x, y)

    win = "Click your jersey, then press any key"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = frame.copy()
        if "xy" in picked:
            cv2.circle(disp, picked["xy"], 10, (0, 255, 0), 2)
        cv2.imshow(win, disp)
        if cv2.waitKey(20) != -1 and "xy" in picked:
            break
    cv2.destroyAllWindows()

    ref = sample_color_ref(frame, picked["xy"])
    ref.to_json(args.out)
    print(f"Saved {args.out}: H={ref.h:.0f} S={ref.s:.0f} V={ref.v:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

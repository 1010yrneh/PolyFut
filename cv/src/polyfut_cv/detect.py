"""YOLOv8 detector: persons + ball, stride, batching, fast frame iteration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from polyfut_cv.geometry import CLASS_BALL, CLASS_PERSON

_MODEL_CACHE: dict[str, object] = {}


@dataclass
class Detection:
    xyxy: np.ndarray
    conf: float
    cls: int


@dataclass
class DetectConfig:
    weights: str = "yolov8n.pt"
    imgsz: int = 640
    conf: float = 0.20
    stride: int = 6
    device: str = "cpu"
    batch_size: int = 8
    classes: tuple[int, ...] = (CLASS_PERSON, CLASS_BALL)
    use_tiled_ball: bool = False  # 2x2 SAHI-style tiles for tiny ball recall


class Detector:
    """Ultralytics wrapper; supports .pt or OpenVINO directory."""

    def __init__(self, cfg: DetectConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        key = f"{cfg.weights}|{cfg.device}"
        if key not in _MODEL_CACHE:
            model = YOLO(cfg.weights)
            if not Path(cfg.weights).is_dir():
                try:
                    model.to(cfg.device)
                except Exception:
                    pass
            _MODEL_CACHE[key] = model
        self.model = _MODEL_CACHE[key]
        self._openvino_imgsz = self._resolve_openvino_imgsz()

    def _resolve_openvino_imgsz(self) -> int | None:
        """OpenVINO exports are fixed-shape (batch=1); default export is 1280."""
        wp = Path(self.cfg.weights)
        if wp.is_dir() and any(wp.glob("*.xml")):
            return 1280
        return None

    def _infer_imgsz(self, imgsz: int | None) -> int:
        ov_sz = self._openvino_imgsz
        if ov_sz is not None:
            return ov_sz
        return imgsz or self.cfg.imgsz

    def _parse_result(self, res) -> tuple[list[Detection], list[Detection]]:
        persons: list[Detection] = []
        balls: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return persons, balls
        xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = res.boxes.conf.cpu().numpy().astype(np.float32)
        cls = res.boxes.cls.cpu().numpy().astype(np.int32)
        for b, c, k in zip(xyxy, conf, cls):
            det = Detection(xyxy=b, conf=float(c), cls=int(k))
            if int(k) == CLASS_PERSON:
                persons.append(det)
            elif int(k) == CLASS_BALL:
                balls.append(det)
        return persons, balls

    def _predict_single_tiled(
        self, frame_bgr: np.ndarray, imgsz: int
    ) -> tuple[list[Detection], list[Detection]]:
        """2x2 overlapping tiles — improves tiny-ball recall on wide shots."""
        h, w = frame_bgr.shape[:2]
        mh, mw = h // 2, w // 2
        tiles = [
            (0, 0, mw, mh),
            (mw, 0, w, mh),
            (0, mh, mw, h),
            (mw, mh, w, h),
        ]
        all_persons: list[Detection] = []
        all_balls: list[Detection] = []
        for x1, y1, x2, y2 in tiles:
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            res = self.model.predict(
                crop, imgsz=imgsz, conf=self.cfg.conf,
                classes=list(self.cfg.classes), verbose=False,
            )[0]
            persons, balls = self._parse_result(res)
            for p in persons:
                b = p.xyxy.copy()
                b[[0, 2]] += x1
                b[[1, 3]] += y1
                all_persons.append(Detection(xyxy=b, conf=p.conf, cls=p.cls))
            for b in balls:
                bb = b.xyxy.copy()
                bb[[0, 2]] += x1
                bb[[1, 3]] += y1
                all_balls.append(Detection(xyxy=bb, conf=b.conf, cls=b.cls))
        if all_balls:
            best = max(all_balls, key=lambda d: d.conf)
            all_balls = [best]
        return all_persons, all_balls

    def predict_frame(
        self, frame_bgr: np.ndarray, imgsz: int | None = None, *, tiled: bool | None = None
    ) -> tuple[list[Detection], list[Detection]]:
        return self.predict_batch([frame_bgr], imgsz=imgsz, tiled=tiled)[0]

    def predict_batch(
        self,
        frames: list[np.ndarray],
        imgsz: int | None = None,
        *,
        tiled: bool | None = None,
    ) -> list[tuple[list[Detection], list[Detection]]]:
        if not frames:
            return []
        sz = self._infer_imgsz(imgsz)
        use_tiled = self.cfg.use_tiled_ball if tiled is None else tiled
        if use_tiled:
            return [self._predict_single_tiled(f, sz) for f in frames]
        # OpenVINO IR is exported batch=1 at fixed imgsz — cannot batch or resize.
        if self._openvino_imgsz is not None:
            out: list[tuple[list[Detection], list[Detection]]] = []
            for frame in frames:
                res = self.model.predict(
                    [frame],
                    imgsz=sz,
                    conf=self.cfg.conf,
                    classes=list(self.cfg.classes),
                    verbose=False,
                )
                out.append(self._parse_result(res[0]))
            return out
        results = self.model.predict(
            frames,
            imgsz=sz,
            conf=self.cfg.conf,
            classes=list(self.cfg.classes),
            verbose=False,
        )
        return [self._parse_result(r) for r in results]


def iter_frames(
    video_path: str | Path,
    *,
    stride: int = 6,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield every `stride`th frame. Skips decode with grab() between kept frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    idx = start_frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        while True:
            if end_frame is not None and idx > end_frame:
                break
            rel = idx - start_frame
            if rel % stride != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def video_info(video_path: str | Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, n, w, h

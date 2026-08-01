"""Soccer-specific ball + player model provisioning.

The default COCO ``yolov8s`` can't see a small soccer ball (≈4% recall on 360p
footage). This wires in a public YOLOv8 model trained on the Roboflow
football-players-detection dataset, which detects the ball and players directly
(≈5x the ball recall). It's a plain weights file on the Hugging Face Hub — no
API key — downloaded on first use.

Source : https://huggingface.co/uisikdag/yolo-v8-football-players-detection
Classes: {0: ball, 1: goalkeeper, 2: player, 3: referee}
License: derived from the Roboflow football-players-detection dataset (CC BY 4.0);
         verify terms for your use.
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

SOCCER_MODEL_URL = (
    "https://huggingface.co/uisikdag/yolo-v8-football-players-detection/"
    "resolve/main/best.pt"
)
SOCCER_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "soccer_uisikdag.pt"
SOCCER_MODEL_OV_DIR = SOCCER_MODEL_PATH.parent / "soccer_uisikdag_openvino_model"
SOCCER_MODEL_IMGSZ = 640  # OpenVINO export is a fixed input size

# Class ids within the soccer model.
SOCCER_BALL_CLASS = 0
SOCCER_GOALKEEPER_CLASS = 1
SOCCER_PLAYER_CLASS = 2
SOCCER_REFEREE_CLASS = 3

# COCO fallback (general yolov8) when the soccer model can't be obtained.
COCO_WEIGHTS = "yolov8s.pt"
COCO_BALL_CLASS = 32   # "sports ball"
COCO_PLAYER_CLASS = 0  # "person"


def ensure_soccer_model(download: bool = True) -> Path | None:
    """Return the local path to the soccer model, downloading it if missing.

    Returns None if it isn't present and can't be fetched (offline) — callers
    should then fall back to COCO.
    """
    if SOCCER_MODEL_PATH.exists() and SOCCER_MODEL_PATH.stat().st_size > 1_000_000:
        return SOCCER_MODEL_PATH
    if not download:
        return None
    try:
        SOCCER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SOCCER_MODEL_PATH.with_suffix(".pt.part")
        urllib.request.urlretrieve(SOCCER_MODEL_URL, tmp)
        tmp.replace(SOCCER_MODEL_PATH)
        return SOCCER_MODEL_PATH
    except Exception:
        return None


_OV_EXPORT_LOCK = threading.Lock()


def _ov_ready() -> bool:
    return SOCCER_MODEL_OV_DIR.is_dir() and any(SOCCER_MODEL_OV_DIR.glob("*.xml"))


def ensure_soccer_model_openvino(download: bool = True) -> Path | None:
    """Return the OpenVINO export of the soccer model (exporting from the .pt on
    first use). OpenVINO is ~6x faster than PyTorch for YOLO on Intel CPUs — the
    difference between an all-day run and a couple of hours. Returns None if it
    can't be produced (offline / openvino not installed); callers fall back to
    the .pt (or COCO).
    """
    if _ov_ready():
        return SOCCER_MODEL_OV_DIR
    pt = ensure_soccer_model(download=download)
    if pt is None:
        return None
    # The async model warm and the seed-clip prefetch can both land here on a
    # fresh install; two concurrent exports to the same directory would corrupt
    # it. First one exports, the second sees _ov_ready() and returns.
    with _OV_EXPORT_LOCK:
        if _ov_ready():
            return SOCCER_MODEL_OV_DIR
        try:
            from ultralytics import YOLO
            YOLO(str(pt)).export(format="openvino", imgsz=SOCCER_MODEL_IMGSZ, half=False)
            return SOCCER_MODEL_OV_DIR if _ov_ready() else None
        except Exception:
            return None

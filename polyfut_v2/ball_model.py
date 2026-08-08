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

import shutil
import sys
import threading
import urllib.request
from pathlib import Path

SOCCER_MODEL_URL = (
    "https://huggingface.co/uisikdag/yolo-v8-football-players-detection/"
    "resolve/main/best.pt"
)


def _models_root() -> Path:
    """Where ``models/`` lives, running from source or from a frozen build.

    In a PyInstaller build the package is unpacked under ``sys._MEIPASS`` and
    ``__file__`` points inside it, so walking up from ``__file__`` happens to
    land in the right place — but only by coincidence of the layout. Reading
    ``_MEIPASS`` says it outright, and it is the difference between finding the
    bundled model and silently falling back to the COCO one that "barely
    detects a soccer ball".

    The installer puts the app in Program Files, which a normal user cannot
    write to, so the model has to be *shipped* rather than fetched on first
    run. Nothing here writes when the bundle is intact.
    """
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base:
            return Path(base) / "models"
        return Path(sys.executable).resolve().parent / "models"
    return Path(__file__).resolve().parent.parent / "models"


SOCCER_MODEL_PATH = _models_root() / "soccer_uisikdag.pt"
SOCCER_MODEL_OV_DIR = SOCCER_MODEL_PATH.parent / "soccer_uisikdag_openvino_model"
SOCCER_MODEL_IMGSZ = 640  # OpenVINO export is a fixed input size

# A second export sized for the ROI pass. The ROI crop is 2*roi_half_px = 240px
# square; feeding it to the 640 export upscales it, and measured on an idle
# i7-1255U that made the "fast" warm path SLOWER than scanning the whole frame:
#
#     full frame 640x360 -> 640 export    134 ms   (medians of 3x96 frames)
#     ROI crop   240x240 -> 640 export    192 ms   <- the supposed shortcut
#     ROI crop   240x240 -> 320 export     64 ms
#
# So the ROI path has been costing 43% MORE than the full scan it replaces, on
# every warm frame of every run. At 320 it is 3x cheaper than it is today and
# 2x cheaper than a full scan, which is what it was always supposed to be.
SOCCER_MODEL_OV_ROI_DIR = SOCCER_MODEL_PATH.parent / "soccer_uisikdag_roi320_openvino_model"
SOCCER_MODEL_IMGSZ_ROI = 320

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


def _ov_roi_ready() -> bool:
    return (SOCCER_MODEL_OV_ROI_DIR.is_dir()
            and any(SOCCER_MODEL_OV_ROI_DIR.glob("*.xml")))


def ensure_soccer_model_openvino_roi(download: bool = True) -> Path | None:
    """The 320-sized export used for ROI crops, or None if it can't be made.

    Exported into a temp directory and moved into place, NOT exported in situ.
    Ultralytics always writes to ``<stem>_openvino_model`` beside the .pt, which
    is the 640 export's own directory — exporting at another size there destroys
    it. (Found the hard way: an in-place 320 export silently deleted the 640
    model this pipeline depends on.) Copying the .pt into a scratch directory
    first means the export lands somewhere disposable and the live 640 model is
    never at risk, even if the export fails halfway.

    Optional by design: without it the ROI pass just uses the 640 model, exactly
    as before, so a failed export costs speed and never correctness.
    """
    if _ov_roi_ready():
        return SOCCER_MODEL_OV_ROI_DIR
    pt = ensure_soccer_model(download=download)
    if pt is None:
        return None
    with _OV_EXPORT_LOCK:
        if _ov_roi_ready():
            return SOCCER_MODEL_OV_ROI_DIR
        tmp_root = SOCCER_MODEL_OV_ROI_DIR.parent / "_roi_export_tmp"
        try:
            from ultralytics import YOLO

            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)
            tmp_root.mkdir(parents=True, exist_ok=True)
            scratch_pt = tmp_root / pt.name
            shutil.copy2(pt, scratch_pt)
            YOLO(str(scratch_pt)).export(
                format="openvino", imgsz=SOCCER_MODEL_IMGSZ_ROI, half=False)
            produced = tmp_root / f"{scratch_pt.stem}_openvino_model"
            if not (produced.is_dir() and any(produced.glob("*.xml"))):
                return None
            if SOCCER_MODEL_OV_ROI_DIR.exists():
                shutil.rmtree(SOCCER_MODEL_OV_ROI_DIR, ignore_errors=True)
            shutil.move(str(produced), str(SOCCER_MODEL_OV_ROI_DIR))
            return SOCCER_MODEL_OV_ROI_DIR if _ov_roi_ready() else None
        except Exception:
            return None
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

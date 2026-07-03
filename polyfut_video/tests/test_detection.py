"""Tests for Stage 5 detection."""

import numpy as np
import pytest

from polyfut_video.pipeline.detection import detect, DetectConfig, Detector


@pytest.mark.slow
def test_detect_on_synthetic_frame():
    """Requires YOLO weights download on first run."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (35, 140, 35)
    det = Detector(DetectConfig(conf_threshold=0.5))
    dets = det.detect_frame(frame)
    assert isinstance(dets, list)
